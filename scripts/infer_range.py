import os
import argparse

import numpy as np
import pandas as pd
import torch

from LGSSM_VAE.config import load_config
from LGSSM_VAE.foundation.errors import LGSSMVAEError
from LGSSM_VAE.foundation.utils import resolve_device
from LGSSM_VAE.data import load_timeseries_frame
from LGSSM_VAE.pipeline import RangeInferOutput, run_range_inference


def _format_time_for_filename(start_str: str, end_str: str) -> str:
    start_dt = pd.to_datetime(start_str, errors="raise")
    end_dt = pd.to_datetime(end_str, errors="raise")
    a = start_dt.strftime("%Y%m%d_%H%M")
    b = end_dt.strftime("%Y%m%d_%H%M")
    return f"{a}_to_{b}"


def _slice_by_timestamps(df: pd.DataFrame, *, ts_col: str, ts: pd.Series) -> pd.DataFrame:
    if df[ts_col].duplicated().any():
        raise SystemExit(f"[错误] CSV 时间戳重复: {ts_col}")
    key = pd.DataFrame({ts_col: ts.astype(str).reset_index(drop=True)})
    out = key.merge(df, on=ts_col, how="left", indicator=True)
    missing = out["_merge"] != "both"
    if bool(missing.any()):
        n_missing = int(missing.sum())
        t0 = str(out.loc[missing, ts_col].iloc[0])
        raise SystemExit(f"[错误] 输入CSV缺少推理输出对应时间戳：missing={n_missing}, first={t0}")
    return out.drop(columns=["_merge"])


def _save_outputs(
    *,
    out_attacked: RangeInferOutput,
    out_normal: RangeInferOutput,
    base_dir: str,
    cfg,
    case: str,
    start_time: str,
    end_time: str,
) -> None:
    os.makedirs(base_dir, exist_ok=True)

    df_attack, ts_col, _ = load_timeseries_frame(str(cfg.infer.attacked_csv))
    df_normal, ts_col_n, _ = load_timeseries_frame(str(cfg.infer.normal_csv))
    if ts_col_n != ts_col:
        raise SystemExit("[错误] infer.normal_csv 与 infer.attacked_csv 时间戳列名不一致")

    ts = out_attacked.timestamps.reset_index(drop=True).astype(str)
    ts_normal = out_normal.timestamps.reset_index(drop=True).astype(str)
    if not ts.equals(ts_normal):
        raise SystemExit("[错误] normal 与 attacked 推理输出时间戳不一致，无法同时导出CSV")

    df_attack_slice = _slice_by_timestamps(df_attack, ts_col=ts_col, ts=ts)
    df_normal_slice = _slice_by_timestamps(df_normal, ts_col=ts_col, ts=ts)
    feature_cols = [c for c in df_attack_slice.columns if c != ts_col]

    normal_infer_path = str(cfg.infer.normal_csv).strip()
    attacked_infer_path = str(cfg.infer.attacked_csv).strip()
    n_base, n_ext = os.path.splitext(os.path.basename(normal_infer_path))
    a_base, a_ext = os.path.splitext(os.path.basename(attacked_infer_path))
    ext = n_ext or a_ext or ".csv"

    tag = _format_time_for_filename(start_time, end_time)
    fallback_normal = f"{case}_normal_clean_{tag}{ext}"
    fallback_attack = f"{case}_attack_clean_{tag}{ext}"
    fallback_normal_repaired = f"{case}_normal_repaired_clean_{tag}{ext}"
    fallback_attack_repaired = f"{case}_attack_repaired_clean_{tag}{ext}"

    normal_name = f"{n_base}{ext}" if n_base else fallback_normal
    attack_name = f"{a_base}{ext}" if a_base else fallback_attack
    if f"{case}_normal_" in n_base:
        normal_repaired_name = f"{n_base.replace(f'{case}_normal_', f'{case}_normal_repaired_', 1)}{ext}"
    elif "_normal_" in n_base:
        normal_repaired_name = f"{n_base.replace('_normal_', '_normal_repaired_', 1)}{ext}"
    else:
        normal_repaired_name = fallback_normal_repaired
    if f"{case}_attack_" in a_base:
        attack_repaired_name = f"{a_base.replace(f'{case}_attack_', f'{case}_attack_repaired_', 1)}{ext}"
    elif "_attack_" in a_base:
        attack_repaired_name = f"{a_base.replace('_attack_', '_attack_repaired_', 1)}{ext}"
    else:
        attack_repaired_name = fallback_attack_repaired

    attacked_normal_path = os.path.join(base_dir, normal_name)
    attacked_attack_path = os.path.join(base_dir, attack_name)
    attacked_repaired_path = os.path.join(base_dir, attack_repaired_name)
    normal_repaired_path = os.path.join(base_dir, normal_repaired_name)

    df_attacked_repair = pd.DataFrame(out_attacked.recon_blend.copy(), columns=feature_cols)
    df_attacked_repair.insert(0, ts_col, ts)
    df_normal_repair = pd.DataFrame(out_normal.recon_blend.copy(), columns=feature_cols)
    df_normal_repair.insert(0, ts_col, ts)

    df_attacked_repair.to_csv(attacked_repaired_path, index=False)
    df_normal_slice.to_csv(attacked_normal_path, index=False)
    df_attack_slice.to_csv(attacked_attack_path, index=False)
    df_normal_repair.to_csv(normal_repaired_path, index=False)

    print(f"  已保存: {attacked_repaired_path}")
    print(f"  已保存: {attacked_normal_path}")
    print(f"  已保存: {attacked_attack_path}")
    print(f"  已保存: {normal_repaired_path}")


def _print_basic_summary(out: RangeInferOutput, *, series: str) -> None:
    """打印基础信息（不输出复杂统计量）"""
    series_name = "正常序列" if series == "normal" else "攻击序列"
    ts = out.timestamps
    N, H = out.recon_blend.shape
    if N == 0:
        print(f"  [警告] {series_name} 无推理输出")
        return
    t_start = str(ts.iloc[0])
    t_end = str(ts.iloc[-1])

    print(f"\n{'─'*60}")
    print(f"  {series_name}")
    print(f"{'─'*60}")
    print(f"  时间范围: {t_start} ~ {t_end}")
    print(f"  数据规模: {N} 步 × {H} 特征")

    if isinstance(out.rmse_vs_clean, np.ndarray):
        valid = int(np.isfinite(out.rmse_vs_clean).sum())
        print(f"  已计算: RMSE(重构/修复后 vs 正常) 每时间步，共 {valid} 步有效")
    if isinstance(out.rmse_obs_vs_clean, np.ndarray) and series == "attacked":
        valid = int(np.isfinite(out.rmse_obs_vs_clean).sum())
        print(f"  已计算: RMSE(攻击前观测 vs 正常) 每时间步，共 {valid} 步有效")


def _print_attacked_metrics(att_out: RangeInferOutput, top_k: int = 20) -> None:
    """攻击序列的精简指标:

    - Repair Worsened Fraction: post-repair RMSE > pre-repair RMSE
    - Top-K Best Repairs: sort by relative improvement (pre - post) / pre
    """
    rmse_before = att_out.rmse_obs_vs_clean
    rmse_repair = att_out.rmse_vs_clean
    if not isinstance(rmse_before, np.ndarray) or not isinstance(rmse_repair, np.ndarray):
        print("  [警告] 攻击序列缺少RMSE向量，无法计算精简指标")
        return
    if rmse_before.size != rmse_repair.size:
        print("  [警告] RMSE向量长度不一致，无法对比")
        return

    before = rmse_before.astype(np.float64)
    repair = rmse_repair.astype(np.float64)
    valid = np.isfinite(before) & np.isfinite(repair)
    n_valid = int(valid.sum())
    if n_valid == 0:
        print("  [警告] 无有效RMSE时间步")
        return

    worse = (repair > before) & valid
    frac_worse = float(worse.sum() / n_valid)
    print(f"\n{'─'*60}")
    print("  Attacked-Series Metrics")
    print(f"{'─'*60}")
    print(
        f"  Repair Worsened Fraction (Post-Repair RMSE > Pre-Repair RMSE): "
        f"{frac_worse:.2%} ({int(worse.sum())}/{n_valid})"
    )

    # Top-K best repairs by relative improvement ratio.
    eps = 1e-12
    ratio = np.full_like(before, np.nan, dtype=np.float64)
    ok = valid & (before > eps)
    ratio[ok] = (before[ok] - repair[ok]) / before[ok]

    finite = np.isfinite(ratio)
    if not np.any(finite):
        print("  [警告] 无有效修复比例可排序")
        return

    order = np.argsort(ratio[finite])[::-1]
    top_idx = np.nonzero(finite)[0][order[: int(top_k)]]

    print()
    print("  Top-20 Best-Repaired Timesteps (sorted by Improvement Ratio):")
    print("  Improvement Ratio = (Pre-Repair RMSE - Post-Repair RMSE) / Pre-Repair RMSE")
    print(f"  {'Rank':>4} {'Timestamp':<18} {'Pre-Repair RMSE':>14} {'Post-Repair RMSE':>15} {'Improvement':>11}")
    print(f"  {'-'*64}")
    ts = att_out.timestamps.reset_index(drop=True)
    for rank, j in enumerate(top_idx, start=1):
        ts_j = str(ts.iloc[int(j)])
        b = float(before[int(j)])
        r = float(repair[int(j)])
        rr = float(ratio[int(j)])
        print(f"  {rank:>4d} {ts_j:<18} {b:>14.4f} {r:>15.4f} {rr:>10.2%}")


def _print_normal_overrepair_metrics(
    out: RangeInferOutput,
    *,
    rmse_threshold: float,
    top_k: int = 20,
) -> None:
    """Normal-series over-repair diagnostics.

    Over-repair here means the repaired (blended) output deviates from the
    original observation on observed entries.
    """
    rmse_change = out.rmse_change_vs_obs
    if not isinstance(rmse_change, np.ndarray):
        print("  [警告] 正常序列缺少 over-repair 向量，无法统计")
        return

    x = rmse_change.astype(np.float64)
    valid = np.isfinite(x)
    n_valid = int(valid.sum())
    if n_valid == 0:
        print("  [警告] 无有效 over-repair 时间步")
        return

    thr = float(rmse_threshold)
    over = valid & (x > thr)
    frac_over = float(over.sum() / n_valid)

    p50 = float(np.nanpercentile(x, 50))
    p90 = float(np.nanpercentile(x, 90))
    p99 = float(np.nanpercentile(x, 99))
    mx = float(np.nanmax(x))

    print(f"\n{'─'*60}")
    print("  Normal-Series Over-Repair Diagnostics")
    print(f"{'─'*60}")
    print(
        "  Change RMSE (Repaired vs Observed) Percentiles: "
        f"p50={p50:.6f}, p90={p90:.6f}, p99={p99:.6f}, max={mx:.6f}"
    )
    if thr > 0.0:
        print(
            f"  Over-Repair Fraction (Change RMSE > {thr:g}): {frac_over:.2%} "
            f"({int(over.sum())}/{n_valid})"
        )

    order = np.argsort(x[valid])[::-1]
    top_idx = np.nonzero(valid)[0][order[: int(top_k)]]

    ts = out.timestamps.reset_index(drop=True)
    w = out.mean_weight_obs
    z = out.mean_abs_change_over_sigma

    print()
    print("  Top-20 Most Over-Repaired Timesteps (sorted by Change RMSE):")
    if isinstance(w, np.ndarray) and isinstance(z, np.ndarray):
        print(
            f"  {'Rank':>4} {'Timestamp':<18} {'Change RMSE':>12} "
            f"{'Mean Obs Weight':>15} {'Mean |Δ|/σ':>12}"
        )
        print(f"  {'-'*68}")
        for rank, j in enumerate(top_idx, start=1):
            ts_j = str(ts.iloc[int(j)])
            print(
                f"  {rank:>4d} {ts_j:<18} {float(x[int(j)]):>12.6f} "
                f"{float(w[int(j)]):>15.4f} {float(z[int(j)]):>12.4f}"
            )
    else:
        print(f"  {'Rank':>4} {'Timestamp':<18} {'Change RMSE':>12}")
        print(f"  {'-'*40}")
        for rank, j in enumerate(top_idx, start=1):
            ts_j = str(ts.iloc[int(j)])
            print(f"  {rank:>4d} {ts_j:<18} {float(x[int(j)]):>12.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LGSSM-VAE 时间范围推理"
    )
    parser.add_argument(
        "--start",
        help='起始时间，如 "2025-09-15 00:00"',
    )
    parser.add_argument(
        "--end",
        help='结束时间，如 "2025-09-20 00:00"',
    )
    parser.add_argument(
        "--overrepair-thr",
        type=float,
        default=0.0,
        help="正常序列过修复阈值：RMSE(修复后 vs 输入观测) > thr 记为过修复；0 表示不做阈值统计",
    )

    args = parser.parse_args()
    cfg = load_config()

    start_time = args.start or str(cfg.infer.start).strip()
    end_time = args.end or str(cfg.infer.end).strip()
    if not start_time or not end_time:
        raise SystemExit("[错误] 请指定 --start 和 --end 时间")

    device = resolve_device(cfg.train.device)
    infer_root = os.path.join("output", cfg.case, "infer")
    save_outputs = bool(cfg.infer.save_outputs)

    series_list = ["normal", "attacked"]

    print(f"\n{'═'*60}")
    print(f"  LGSSM-VAE 推理")
    print(f"{'═'*60}")
    print(f"  案例: {cfg.case}")
    print(f"  时间: {start_time} ~ {end_time}")
    print("  序列: normal + attacked")

    outputs: dict[str, RangeInferOutput] = {}

    for s in series_list:
        out = run_range_inference(
            cfg,
            series=s,
            start_time=start_time,
            end_time=end_time,
            device=device,
        )
        _print_basic_summary(out, series=s)
        outputs[s] = out

    if save_outputs:
        if "normal" not in outputs or "attacked" not in outputs:
            raise SystemExit("[错误] 推理输出缺失，无法导出CSV")
        _save_outputs(
            out_attacked=outputs["attacked"],
            out_normal=outputs["normal"],
            base_dir=infer_root,
            cfg=cfg,
            case=str(cfg.case),
            start_time=start_time,
            end_time=end_time,
        )

    # Print simplified attacked-only comparison metrics.
    if "attacked" in outputs:
        _print_attacked_metrics(outputs["attacked"], top_k=20)
    if "normal" in outputs:
        _print_normal_overrepair_metrics(
            outputs["normal"],
            rmse_threshold=float(args.overrepair_thr),
            top_k=20,
        )


if __name__ == "__main__":
    try:
        main()
    except LGSSMVAEError as e:
        raise SystemExit(str(e)) from None
