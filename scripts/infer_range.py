import os
import argparse

import numpy as np
import pandas as pd
import torch

from tcn_vae.config import load_config
from tcn_vae.inference import run_range_inference, RangeInferOutput
from tcn_vae.utils import resolve_device


def _format_time_for_filename(start_str: str, end_str: str) -> str:
    start_dt = pd.to_datetime(start_str, errors="raise")
    end_dt = pd.to_datetime(end_str, errors="raise")
    a = start_dt.strftime("%Y%m%d_%H%M")
    b = end_dt.strftime("%Y%m%d_%H%M")
    return f"{a}_to_{b}"


def _save_range_outputs(
    out: RangeInferOutput,
    *,
    base_dir: str,
    case: str,
    series: str,
    start_time: str,
    end_time: str,
) -> None:
    os.makedirs(base_dir, exist_ok=True)
    tag = _format_time_for_filename(start_time, end_time)
    series = series.lower().strip()

    # Reconstructed (blended) series
    recon_path = os.path.join(
        base_dir,
        f"{case}_{series}_recon_{tag}.csv",
    )

    ts = out.timestamps.reset_index(drop=True)
    N, H = out.recon_blend.shape

    # Reconstructed series: timestamp + H features
    df_recon = pd.DataFrame(out.recon_blend.copy())
    df_recon.insert(0, "timestamp", ts.astype(str))
    df_recon.to_csv(recon_path, index=False)

    # Optional per-step RMSE vs clean (for normal/attacked with ground truth)
    if out.rmse_vs_clean is not None:
        rmse_path = os.path.join(
            base_dir,
            f"{case}_{series}_rmse_{tag}.tsv",
        )
        idx = np.arange(N, dtype=np.int64)
        data = {
            "idx": idx,
            "timestamp": ts.astype(str),
            "rmse_repair": out.rmse_vs_clean.astype(np.float32),
        }
        if out.rmse_obs_vs_clean is not None:
            data["rmse_obs"] = out.rmse_obs_vs_clean.astype(np.float32)
        df_rmse = pd.DataFrame(data)
        df_rmse.to_csv(rmse_path, sep="\t", index=False)

    print(f"  已保存: {recon_path}")
    if out.rmse_vs_clean is not None:
        print(f"  已保存: {rmse_path}")


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

    - 修复变差比例: RMSE_repair > RMSE_before 的时间步比例
    - Top-K 最佳修复: 按修复比例 (before-after)/before 排序
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
    print("  攻击序列对比指标")
    print(f"{'─'*60}")
    print(
        f"  修复变差比例 (RMSE_修复 > RMSE_攻击前): "
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
    print("  前20个修复效果最好的时间步（按修复比例从高到低）:")
    print("  修复比例 = (RMSE_攻击前 - RMSE_修复后) / RMSE_攻击前")
    print(f"  {'序号':>4} {'时间戳':<18} {'攻击前RMSE':>12} {'修复后RMSE':>12} {'修复比例':>10}")
    print(f"  {'-'*64}")
    ts = att_out.timestamps.reset_index(drop=True)
    for rank, j in enumerate(top_idx, start=1):
        ts_j = str(ts.iloc[int(j)])
        b = float(before[int(j)])
        r = float(repair[int(j)])
        rr = float(ratio[int(j)])
        print(f"  {rank:>4d} {ts_j:<18} {b:>12.4f} {r:>12.4f} {rr:>9.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TCN-VAE 时间范围推理"
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
        "--series",
        choices=["normal", "attacked", "both"],
        help="推理序列: normal(正常), attacked(攻击), both(两者)",
    )

    args = parser.parse_args()
    cfg = load_config()

    start_time = args.start or getattr(cfg, "infer_start", "").strip()
    end_time = args.end or getattr(cfg, "infer_end", "").strip()
    series_cfg = getattr(cfg, "infer_series", "both")
    series_arg = args.series or series_cfg
    series_opt = str(series_arg).strip().lower() if series_arg is not None else "both"

    if series_opt not in ("normal", "attacked", "both"):
        raise SystemExit(f"[错误] 无效序列 '{series_opt}'，可选: normal, attacked, both")
    if not start_time or not end_time:
        raise SystemExit("[错误] 请指定 --start 和 --end 时间")

    device = resolve_device(getattr(cfg, "device", "cpu"))
    infer_root = os.path.join("output", cfg.case, "infer")
    save_outputs = bool(getattr(cfg, "infer_save_outputs", False))

    series_list = ["normal", "attacked"] if series_opt == "both" else [series_opt]

    print(f"\n{'═'*60}")
    print(f"  TCN-VAE 推理")
    print(f"{'═'*60}")
    print(f"  案例: {cfg.case}")
    print(f"  时间: {start_time} ~ {end_time}")
    print(f"  序列: {series_opt}")

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
            _save_range_outputs(
                out,
                base_dir=infer_root,
                case=str(cfg.case),
                series=s,
                start_time=start_time,
                end_time=end_time,
            )

    # Print simplified attacked-only comparison metrics.
    if "attacked" in outputs:
        _print_attacked_metrics(outputs["attacked"], top_k=20)


if __name__ == "__main__":
    main()
