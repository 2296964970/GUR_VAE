"""训练数据质量检查脚本。

检查项目:
1. 基本统计信息（样本数、特征数、缺失值）
2. Persistence baseline（用 x[t-1] 预测 x[t] 的 MSE）
3. 时序自相关分析
4. 标准化后的分布检查
5. 特征方差分布
6. 平稳性检查
7. 周期性分析（日/周周期）
8. 问题特征识别
9. 多步预测 baseline
10. 窗口级统计
11. 信噪比估计

用法:
    python scripts/check_data_quality.py [--save-plots]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tcn_vae.config import load_config
from tcn_vae.data import (
    times_to_5min_index,
    masked_robust_slot_stats,
    apply_standardization_slotwise,
)


def load_data(csv_path: str) -> pd.DataFrame:
    """加载 CSV 数据。"""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    return df


def standardize_like_training(
    x_raw: np.ndarray,
    ts: pd.Series,
    *,
    train_ratio: float,
    clip_k: float,
    std_floor: float,
) -> np.ndarray:
    """按训练同样的 robust 标准化方式得到 x_std。"""
    if x_raw.ndim != 2:
        raise ValueError("x_raw must be [T,H]")
    T = x_raw.shape[0]
    m = (~np.isnan(x_raw)).astype(np.float32)
    x = np.nan_to_num(x_raw, nan=0.0).astype(np.float32)
    s_train = slice(0, int(T * float(train_ratio)))

    # Keep the original project behaviour: 5-minute slotwise robust standardization.
    slots = times_to_5min_index(ts)
    slot_count = 288

    slot_mean, slot_std = masked_robust_slot_stats(
        x[s_train],
        m[s_train],
        slots[s_train],
        slot_count=int(slot_count),
        std_floor=float(std_floor),
    )
    return apply_standardization_slotwise(
        x, m, slots, slot_mean, slot_std, clip_k=float(clip_k)
    )


def check_basic_stats(df: pd.DataFrame) -> dict:
    """基本统计信息。"""
    print("\n" + "=" * 60)
    print("【1】基本统计信息")
    print("=" * 60)

    n_samples, n_features = df.shape
    n_missing = df.isna().sum().sum()
    missing_ratio = n_missing / (n_samples * n_features) * 100

    print(f"  样本数 (时间步): {n_samples}")
    print(f"  特征数: {n_features}")
    print(f"  缺失值: {n_missing} ({missing_ratio:.2f}%)")
    print(f"  时间范围: {df.index[0]} ~ {df.index[-1]}")

    # 时间间隔检查
    if len(df) > 1:
        time_diffs = pd.Series(df.index).diff().dropna()
        most_common_diff = time_diffs.mode().iloc[0] if len(time_diffs) > 0 else None
        print(f"  最常见时间间隔: {most_common_diff}")

        irregular_count = (time_diffs != most_common_diff).sum()
        if irregular_count > 0:
            print(f"  ⚠️  不规则时间间隔数: {irregular_count}")

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "missing_ratio": missing_ratio,
    }


def check_persistence_baseline(x: np.ndarray, x_std: np.ndarray) -> dict:
    """Persistence baseline: 用 x[t-1] 预测 x[t]。"""
    print("\n" + "=" * 60)
    print("【2】Persistence Baseline (用上一步预测当前步)")
    print("=" * 60)

    # 原始数据
    diff_raw = x[1:] - x[:-1]
    mse_raw = np.nanmean(diff_raw ** 2)
    rmse_raw = np.sqrt(mse_raw)

    # 标准化数据
    diff_std = x_std[1:] - x_std[:-1]
    mse_std = np.nanmean(diff_std ** 2)
    rmse_std = np.sqrt(mse_std)

    print(f"  原始数据:")
    print(f"    MSE:  {mse_raw:.6f}")
    print(f"    RMSE: {rmse_raw:.6f}")
    print(f"  标准化数据:")
    print(f"    MSE:  {mse_std:.6f}")
    print(f"    RMSE: {rmse_std:.6f}")

    print(f"\n【解读】")
    if mse_std > 1.0:
        print(f"  ⚠️  标准化 MSE > 1.0，说明相邻时间步变化较大")
        print(f"     如果模型 MSE ≈ {mse_std:.2f}，可能已接近 persistence baseline")
    elif mse_std > 0.5:
        print(f"  📊 标准化 MSE ≈ {mse_std:.2f}，数据有一定变化但可预测")
    else:
        print(f"  ✓  标准化 MSE < 0.5，数据时序平滑，应该容易预测")

    return {
        "persistence_mse_raw": mse_raw,
        "persistence_mse_std": mse_std,
    }


def check_autocorrelation(x_std: np.ndarray, max_lag: int = 10) -> dict:
    """时序自相关分析。"""
    print("\n" + "=" * 60)
    print("【3】时序自相关分析")
    print("=" * 60)

    n_samples, n_features = x_std.shape

    # 计算各 lag 的平均自相关
    autocorrs = {}
    for lag in [1, 2, 5, 10]:
        if lag >= n_samples:
            continue
        corrs = []
        for i in range(n_features):
            col = x_std[:, i]
            valid = ~np.isnan(col)
            if valid.sum() > lag + 10:
                col_clean = col[valid]
                if len(col_clean) > lag:
                    corr = np.corrcoef(col_clean[:-lag], col_clean[lag:])[0, 1]
                    if not np.isnan(corr):
                        corrs.append(corr)
        if corrs:
            autocorrs[lag] = np.mean(corrs)

    print(f"  平均自相关系数:")
    for lag, corr in autocorrs.items():
        bar = "█" * int(abs(corr) * 20)
        print(f"    lag={lag:2d}: {corr:+.4f} {bar}")

    lag1_corr = autocorrs.get(1, 0)
    print(f"\n【解读】")
    if lag1_corr > 0.9:
        print(f"  ✓  lag-1 自相关 > 0.9，数据高度平滑，时序模型应该有效")
    elif lag1_corr > 0.7:
        print(f"  ✓  lag-1 自相关 > 0.7，数据有较好的时序结构")
    elif lag1_corr > 0.5:
        print(f"  📊 lag-1 自相关 ≈ {lag1_corr:.2f}，时序结构中等")
    else:
        print(f"  ⚠️  lag-1 自相关 < 0.5，时序结构弱，可能难以用时序模型预测")

    return {"autocorr_lag1": lag1_corr, "autocorrs": autocorrs}


def check_distribution(x: np.ndarray, x_std: np.ndarray) -> dict:
    """分布检查。"""
    print("\n" + "=" * 60)
    print("【4】数据分布检查")
    print("=" * 60)

    # 原始数据统计
    print(f"  原始数据:")
    print(f"    均值范围: [{np.nanmean(x, axis=0).min():.4f}, {np.nanmean(x, axis=0).max():.4f}]")
    print(f"    标准差范围: [{np.nanstd(x, axis=0).min():.4f}, {np.nanstd(x, axis=0).max():.4f}]")

    # 标准化后
    print(f"  标准化后:")
    std_mean = np.nanmean(x_std, axis=0)
    std_std = np.nanstd(x_std, axis=0)
    print(f"    均值范围: [{std_mean.min():.6f}, {std_mean.max():.6f}] (应接近 0)")
    print(f"    标准差范围: [{std_std.min():.6f}, {std_std.max():.6f}] (应接近 1)")

    # 检查极端值
    outlier_ratio = np.mean(np.abs(x_std) > 3) * 100
    extreme_ratio = np.mean(np.abs(x_std) > 5) * 100

    print(f"  异常值比例:")
    print(f"    |z| > 3: {outlier_ratio:.2f}%")
    print(f"    |z| > 5: {extreme_ratio:.2f}%")

    if extreme_ratio > 1:
        print(f"  ⚠️  极端值较多 (|z|>5 占 {extreme_ratio:.2f}%)，考虑使用 clip")

    return {
        "outlier_ratio_3sigma": outlier_ratio,
        "outlier_ratio_5sigma": extreme_ratio,
    }


def check_feature_variance(x: np.ndarray) -> dict:
    """特征方差分析。"""
    print("\n" + "=" * 60)
    print("【5】特征方差分析")
    print("=" * 60)

    variances = np.nanvar(x, axis=0)

    print(f"  方差统计:")
    print(f"    最小: {variances.min():.6f}")
    print(f"    最大: {variances.max():.6f}")
    print(f"    均值: {variances.mean():.6f}")
    print(f"    中位数: {np.median(variances):.6f}")

    # 检查方差极端的特征
    very_low_var = np.sum(variances < 1e-6)
    very_high_var = np.sum(variances > 1e6)

    if very_low_var > 0:
        print(f"  ⚠️  {very_low_var} 个特征方差极低 (< 1e-6)，可能是常数列")
    if very_high_var > 0:
        print(f"  ⚠️  {very_high_var} 个特征方差极高 (> 1e6)，可能需要特殊处理")

    # 方差比值
    var_ratio = variances.max() / (variances.min() + 1e-10)
    print(f"  方差比值 (max/min): {var_ratio:.2e}")
    if var_ratio > 1e6:
        print(f"  ⚠️  方差差异过大，标准化很重要")

    return {
        "var_min": variances.min(),
        "var_max": variances.max(),
        "var_ratio": var_ratio,
        "low_var_features": very_low_var,
    }


def check_stationarity(x_std: np.ndarray, n_segments: int = 4) -> dict:
    """简单平稳性检查（分段统计）。"""
    print("\n" + "=" * 60)
    print("【6】平稳性检查 (分段统计)")
    print("=" * 60)

    n_samples = len(x_std)
    seg_size = n_samples // n_segments

    seg_means = []
    seg_stds = []

    for i in range(n_segments):
        start = i * seg_size
        end = start + seg_size if i < n_segments - 1 else n_samples
        seg = x_std[start:end]
        seg_means.append(np.nanmean(seg))
        seg_stds.append(np.nanstd(seg))

    print(f"  分段均值: {[f'{m:.4f}' for m in seg_means]}")
    print(f"  分段标准差: {[f'{s:.4f}' for s in seg_stds]}")

    mean_drift = max(seg_means) - min(seg_means)
    std_drift = max(seg_stds) - min(seg_stds)

    print(f"\n  均值漂移: {mean_drift:.4f}")
    print(f"  标准差漂移: {std_drift:.4f}")

    if mean_drift > 0.5 or std_drift > 0.3:
        print(f"  ⚠️  数据可能非平稳，不同时间段统计特性有差异")
    else:
        print(f"  ✓  数据基本平稳")

    return {
        "mean_drift": mean_drift,
        "std_drift": std_drift,
    }


def compute_model_vs_baseline(persistence_mse: float, model_mse: float = 1.17):
    """对比模型 MSE 和 baseline。"""
    print("\n" + "=" * 60)
    print("【7】模型 vs Baseline 对比")
    print("=" * 60)

    print(f"  Persistence baseline MSE: {persistence_mse:.4f}")
    print(f"  当前模型 MSE (参考值): {model_mse:.4f}")

    if model_mse < persistence_mse * 0.9:
        improvement = (1 - model_mse / persistence_mse) * 100
        print(f"  ✓  模型优于 baseline {improvement:.1f}%")
    elif model_mse < persistence_mse * 1.1:
        print(f"  📊 模型与 baseline 接近，可能已达数据可预测性极限")
    else:
        print(f"  ⚠️  模型比 baseline 差，可能存在训练问题")


def check_periodicity(df: pd.DataFrame, x_std: np.ndarray) -> dict:
    """周期性分析（日周期、周周期）。"""
    print("\n" + "=" * 60)
    print("【8】周期性分析")
    print("=" * 60)

    results = {}

    # 检查是否有时间索引
    if not isinstance(df.index, pd.DatetimeIndex):
        print("  ⚠️  无法分析周期性：索引不是时间格式")
        return results

    # 提取时间特征
    hours = df.index.hour
    days_of_week = df.index.dayofweek

    # 日内周期分析（按小时分组）
    print("\n  【日内周期】(按小时)")
    hourly_means = []
    hourly_stds = []
    for h in range(24):
        mask = hours == h
        if mask.sum() > 0:
            hourly_means.append(np.nanmean(x_std[mask]))
            hourly_stds.append(np.nanstd(x_std[mask]))

    if hourly_means:
        hourly_range = max(hourly_means) - min(hourly_means)
        results["hourly_mean_range"] = hourly_range
        print(f"    小时均值范围: {hourly_range:.4f}")

        # 找出高/低峰时段
        peak_hour = np.argmax(hourly_means)
        trough_hour = np.argmin(hourly_means)
        print(f"    高峰时段: {peak_hour}:00 (均值 {hourly_means[peak_hour]:+.4f})")
        print(f"    低谷时段: {trough_hour}:00 (均值 {hourly_means[trough_hour]:+.4f})")

        if hourly_range > 0.3:
            print(f"    ✓  存在明显日内周期")
        else:
            print(f"    📊 日内周期不明显")

    # 周周期分析（按星期几分组）
    print("\n  【周周期】(按星期)")
    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekly_means = []
    for d in range(7):
        mask = days_of_week == d
        if mask.sum() > 0:
            weekly_means.append(np.nanmean(x_std[mask]))

    if weekly_means:
        weekly_range = max(weekly_means) - min(weekly_means)
        results["weekly_mean_range"] = weekly_range
        print(f"    星期均值范围: {weekly_range:.4f}")

        peak_day = np.argmax(weekly_means)
        trough_day = np.argmin(weekly_means)
        print(f"    高峰日: {weekday_names[peak_day]} (均值 {weekly_means[peak_day]:+.4f})")
        print(f"    低谷日: {weekday_names[trough_day]} (均值 {weekly_means[trough_day]:+.4f})")

        if weekly_range > 0.2:
            print(f"    ✓  存在明显周周期")
        else:
            print(f"    📊 周周期不明显")

    return results


def identify_problem_features(df: pd.DataFrame, x: np.ndarray, x_std: np.ndarray) -> dict:
    """识别问题特征。"""
    print("\n" + "=" * 60)
    print("【9】问题特征识别")
    print("=" * 60)

    n_features = x.shape[1]
    feature_names = df.columns.tolist()

    # 计算各种特征指标
    variances = np.nanvar(x, axis=0)
    means = np.nanmean(x, axis=0)

    # 计算每个特征的 lag-1 自相关
    autocorrs = []
    for i in range(n_features):
        col = x_std[:, i]
        valid = ~np.isnan(col)
        if valid.sum() > 10:
            col_clean = col[valid]
            corr = np.corrcoef(col_clean[:-1], col_clean[1:])[0, 1]
            autocorrs.append(corr if not np.isnan(corr) else 0)
        else:
            autocorrs.append(0)
    autocorrs = np.array(autocorrs)

    # 常数特征（方差极低）
    const_mask = variances < 1e-6
    const_features = np.where(const_mask)[0]

    print(f"\n  【常数/近常数特征】(方差 < 1e-6)")
    if len(const_features) > 0:
        print(f"    数量: {len(const_features)}")
        for idx in const_features[:5]:  # 最多显示5个
            print(f"      - {feature_names[idx]}: var={variances[idx]:.2e}, mean={means[idx]:.4f}")
        if len(const_features) > 5:
            print(f"      ... 还有 {len(const_features) - 5} 个")
    else:
        print(f"    无")

    # 低自相关特征（难以预测）
    low_autocorr_mask = autocorrs < 0.3
    low_autocorr_features = np.where(low_autocorr_mask)[0]

    print(f"\n  【低自相关特征】(lag-1 < 0.3，难以时序预测)")
    print(f"    数量: {len(low_autocorr_features)} / {n_features} ({100*len(low_autocorr_features)/n_features:.1f}%)")

    # 最难预测的5个特征
    worst_5 = np.argsort(autocorrs)[:5]
    print(f"    最难预测的特征:")
    for idx in worst_5:
        print(f"      - {feature_names[idx]}: autocorr={autocorrs[idx]:.4f}")

    # 高自相关特征（容易预测）
    high_autocorr_mask = autocorrs > 0.8
    high_autocorr_features = np.where(high_autocorr_mask)[0]

    print(f"\n  【高自相关特征】(lag-1 > 0.8，容易时序预测)")
    print(f"    数量: {len(high_autocorr_features)} / {n_features} ({100*len(high_autocorr_features)/n_features:.1f}%)")

    # 最容易预测的5个特征
    best_5 = np.argsort(autocorrs)[-5:][::-1]
    print(f"    最容易预测的特征:")
    for idx in best_5:
        print(f"      - {feature_names[idx]}: autocorr={autocorrs[idx]:.4f}")

    # 异常值多的特征
    outlier_ratios = np.mean(np.abs(x_std) > 3, axis=0)
    high_outlier_mask = outlier_ratios > 0.01  # >1% 异常值
    high_outlier_features = np.where(high_outlier_mask)[0]

    print(f"\n  【高异常值特征】(|z|>3 占比 > 1%)")
    print(f"    数量: {len(high_outlier_features)}")
    if len(high_outlier_features) > 0:
        worst_outlier = np.argsort(outlier_ratios)[-3:][::-1]
        for idx in worst_outlier:
            if outlier_ratios[idx] > 0.01:
                print(f"      - {feature_names[idx]}: {100*outlier_ratios[idx]:.2f}% 异常值")

    return {
        "const_features": len(const_features),
        "low_autocorr_features": len(low_autocorr_features),
        "high_autocorr_features": len(high_autocorr_features),
        "high_outlier_features": len(high_outlier_features),
        "autocorrs": autocorrs,
    }


def check_multi_step_baseline(x_std: np.ndarray) -> dict:
    """多步预测 baseline 分析。"""
    print("\n" + "=" * 60)
    print("【10】多步预测 Baseline")
    print("=" * 60)

    lags = [1, 2, 3, 6, 12, 24, 48, 96]  # 5分钟数据：对应5分钟到8小时
    mse_by_lag = {}

    print(f"  用 x[t-lag] 预测 x[t] 的 MSE:")
    print(f"  {'Lag':>6} | {'时间间隔':>10} | {'MSE':>8} | {'RMSE':>8} | 图示")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+{'-'*20}")

    for lag in lags:
        if lag >= len(x_std):
            continue
        diff = x_std[lag:] - x_std[:-lag]
        mse = np.nanmean(diff ** 2)
        rmse = np.sqrt(mse)
        mse_by_lag[lag] = mse

        # 计算时间间隔（假设5分钟采样）
        minutes = lag * 5
        if minutes < 60:
            time_str = f"{minutes}分钟"
        else:
            time_str = f"{minutes//60}小时{minutes%60:02d}分" if minutes % 60 else f"{minutes//60}小时"

        bar = "█" * min(int(mse * 10), 20)
        print(f"  {lag:>6} | {time_str:>10} | {mse:>8.4f} | {rmse:>8.4f} | {bar}")

    print(f"\n【解读】")
    if mse_by_lag:
        mse_1 = mse_by_lag.get(1, 1)
        mse_96 = mse_by_lag.get(96, mse_by_lag.get(max(mse_by_lag.keys()), 2))

        if mse_96 / mse_1 < 1.5:
            print(f"  ⚠️  长期预测 MSE 增长缓慢，说明数据本身就像随机游走")
        else:
            print(f"  📊 MSE 随 lag 增加而增长，时序结构存在")

    return {"mse_by_lag": mse_by_lag}


def check_window_statistics(x_std: np.ndarray, window_size: int = 96, stride: int = 24) -> dict:
    """窗口级统计（与模型训练一致）。"""
    print("\n" + "=" * 60)
    print(f"【11】窗口级统计 (window={window_size}, stride={stride})")
    print("=" * 60)

    n_samples = len(x_std)
    n_windows = (n_samples - window_size) // stride + 1

    window_means = []
    window_stds = []
    window_ranges = []  # 窗口内变化范围

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        window = x_std[start:end]

        window_means.append(np.nanmean(window))
        window_stds.append(np.nanstd(window))
        # 窗口内最大变化
        window_ranges.append(np.nanmax(window) - np.nanmin(window))

    window_means = np.array(window_means)
    window_stds = np.array(window_stds)
    window_ranges = np.array(window_ranges)

    print(f"  窗口数量: {n_windows}")
    print(f"\n  窗口均值统计:")
    print(f"    mean: {window_means.mean():.4f}")
    print(f"    std:  {window_means.std():.4f}")
    print(f"    范围: [{window_means.min():.4f}, {window_means.max():.4f}]")

    print(f"\n  窗口标准差统计:")
    print(f"    mean: {window_stds.mean():.4f}")
    print(f"    std:  {window_stds.std():.4f}")
    print(f"    范围: [{window_stds.min():.4f}, {window_stds.max():.4f}]")

    print(f"\n  窗口内变化范围 (max-min):")
    print(f"    mean: {window_ranges.mean():.4f}")
    print(f"    std:  {window_ranges.std():.4f}")
    print(f"    范围: [{window_ranges.min():.4f}, {window_ranges.max():.4f}]")

    # 窗口间差异
    print(f"\n【解读】")
    if window_means.std() > 0.3:
        print(f"  ⚠️  窗口间均值差异大 (std={window_means.std():.4f})，数据可能非平稳")
    else:
        print(f"  ✓  窗口间均值一致性好")

    if window_ranges.mean() > 4:
        print(f"  📊 窗口内变化范围大 (mean={window_ranges.mean():.2f})，重建难度高")

    return {
        "n_windows": n_windows,
        "window_mean_std": window_means.std(),
        "window_range_mean": window_ranges.mean(),
    }


def estimate_snr(x: np.ndarray, x_std: np.ndarray) -> dict:
    """信噪比估计（基于平滑度）。"""
    print("\n" + "=" * 60)
    print("【12】信噪比估计")
    print("=" * 60)

    # 方法1: 基于一阶差分的噪声估计 (MAD)
    diff = np.diff(x_std, axis=0)
    noise_est_mad = np.median(np.abs(diff)) / 0.6745  # MAD to std

    # 方法2: 基于局部平滑的噪声估计
    # 简单移动平均作为信号估计 (用 numpy 实现)
    kernel_size = 5

    def moving_average(arr, window):
        """简单移动平均实现"""
        ret = np.cumsum(arr, axis=0, dtype=float)
        ret[window:] = ret[window:] - ret[:-window]
        result = np.zeros_like(arr)
        # 处理边界
        for i in range(window):
            result[i] = np.mean(arr[:i+1], axis=0)
        result[window-1:] = ret[window-1:] / window
        return result

    x_smooth = moving_average(x_std, kernel_size)
    residual = x_std - x_smooth
    noise_est_smooth = np.std(residual)

    # 信号方差 (总方差 - 噪声方差)
    total_var = np.var(x_std)
    signal_var_mad = max(total_var - noise_est_mad**2, 0.01)
    signal_var_smooth = max(total_var - noise_est_smooth**2, 0.01)

    snr_mad = 10 * np.log10(signal_var_mad / (noise_est_mad**2 + 1e-10))
    snr_smooth = 10 * np.log10(signal_var_smooth / (noise_est_smooth**2 + 1e-10))

    print(f"  方法1 (基于一阶差分 MAD):")
    print(f"    噪声 std 估计: {noise_est_mad:.4f}")
    print(f"    SNR 估计: {snr_mad:.2f} dB")

    print(f"\n  方法2 (基于局部平滑残差):")
    print(f"    噪声 std 估计: {noise_est_smooth:.4f}")
    print(f"    SNR 估计: {snr_smooth:.2f} dB")

    print(f"\n【解读】")
    avg_snr = (snr_mad + snr_smooth) / 2
    if avg_snr < 0:
        print(f"  ⚠️  SNR < 0 dB，数据噪声占主导，预测极其困难")
    elif avg_snr < 5:
        print(f"  ⚠️  SNR < 5 dB，噪声较大，这可能是 MSE 难以降低的主因")
    elif avg_snr < 10:
        print(f"  📊 SNR ≈ {avg_snr:.1f} dB，中等噪声水平")
    else:
        print(f"  ✓  SNR > 10 dB，信号质量较好")

    # 理论最优 MSE 估计
    theoretical_min_mse = noise_est_smooth ** 2
    print(f"\n  理论最优 MSE (如果能完美预测信号): ~{theoretical_min_mse:.4f}")
    print(f"  当前模型 MSE ≈ 1.17，差距: {1.17 - theoretical_min_mse:.4f}")

    return {
        "noise_std_mad": noise_est_mad,
        "noise_std_smooth": noise_est_smooth,
        "snr_mad": snr_mad,
        "snr_smooth": snr_smooth,
        "theoretical_min_mse": theoretical_min_mse,
    }


def save_diagnostic_plots(df: pd.DataFrame, x_std: np.ndarray, autocorrs: np.ndarray, output_dir: str = "output/diagnostics"):
    """保存诊断图表。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️  matplotlib 未安装，跳过图表生成")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("【13】生成诊断图表")
    print("=" * 60)

    # 图1: 自相关分布直方图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.hist(autocorrs, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0.5, color='r', linestyle='--', label='threshold=0.5')
    ax.set_xlabel('Lag-1 Autocorrelation')
    ax.set_ylabel('Feature Count')
    ax.set_title('Feature Autocorrelation Distribution')
    ax.legend()

    # 图2: 特征方差分布
    ax = axes[0, 1]
    variances = np.nanvar(df.values, axis=0)
    ax.hist(np.log10(variances + 1e-10), bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('log10(Variance)')
    ax.set_ylabel('Feature Count')
    ax.set_title('Feature Variance Distribution (log scale)')

    # 图3: 时序样例（前3个特征）
    ax = axes[1, 0]
    n_plot = min(500, len(x_std))
    for i in range(min(3, x_std.shape[1])):
        ax.plot(x_std[:n_plot, i], alpha=0.7, label=f'Feature {i}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Standardized Value')
    ax.set_title('Sample Time Series (first 3 features)')
    ax.legend()

    # 图4: 多步 MSE baseline
    ax = axes[1, 1]
    lags = [1, 2, 3, 6, 12, 24, 48, 96]
    mses = []
    for lag in lags:
        if lag < len(x_std):
            diff = x_std[lag:] - x_std[:-lag]
            mses.append(np.nanmean(diff ** 2))
    ax.plot(lags[:len(mses)], mses, 'o-', linewidth=2, markersize=8)
    ax.axhline(y=1.17, color='r', linestyle='--', label='Model MSE ≈ 1.17')
    ax.set_xlabel('Lag (time steps)')
    ax.set_ylabel('MSE')
    ax.set_title('Multi-step Prediction Baseline')
    ax.legend()

    plt.tight_layout()
    plot_path = f"{output_dir}/data_quality_diagnosis.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  ✓  图表已保存: {plot_path}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="训练数据质量检查")
    parser.add_argument("--save-plots", action="store_true", help="保存诊断图表")
    parser.add_argument("--model-mse", type=float, default=1.17, help="当前模型的 MSE (用于对比)")
    cli_args, _ = parser.parse_known_args()

    args = load_config()

    print("=" * 60)
    print("训练数据质量检查 (完整版)")
    print("=" * 60)

    csv_path = args.train_normal_csv
    print(f"数据文件: {csv_path}")

    if not Path(csv_path).exists():
        print(f"错误: 文件不存在 {csv_path}")
        sys.exit(1)

    # 加载数据
    df = load_data(csv_path)
    x = df.values.astype(np.float64)

    # 标准化：与训练一致，固定为 5-minute slotwise robust 标准化。
    x_std = standardize_like_training(
        x,
        df.index.to_series(),
        train_ratio=getattr(args, "train_ratio", 0.7),
        clip_k=getattr(args, "clip_k", 0.0),
        std_floor=getattr(args, "std_floor", 1e-3),
    )
    print("[info] 使用训练式标准化: slotwise-5min")

    # 获取窗口配置
    window_size = getattr(args, 'time_length', 96)
    stride = getattr(args, 'stride', 24)

    # ========== 基础检查 ==========
    basic = check_basic_stats(df)
    persistence = check_persistence_baseline(x, x_std)
    autocorr = check_autocorrelation(x_std)
    dist = check_distribution(x, x_std)
    variance = check_feature_variance(x)
    stationarity = check_stationarity(x_std)

    # 模型对比
    compute_model_vs_baseline(persistence["persistence_mse_std"], cli_args.model_mse)

    # ========== 扩展检查 ==========
    periodicity = check_periodicity(df, x_std)
    problem_features = identify_problem_features(df, x, x_std)
    multi_step = check_multi_step_baseline(x_std)
    window_stats = check_window_statistics(x_std, window_size, stride)
    snr = estimate_snr(x, x_std)

    # 保存图表
    if cli_args.save_plots:
        save_diagnostic_plots(df, x_std, problem_features["autocorrs"])

    # ========== 总结 ==========
    print("\n" + "=" * 60)
    print("【综合诊断总结】")
    print("=" * 60)

    issues = []
    suggestions = []

    # 检查各项指标
    if persistence["persistence_mse_std"] > 1.0:
        issues.append(f"相邻时间步变化大 (persistence MSE = {persistence['persistence_mse_std']:.3f})")
        suggestions.append("数据本身变化剧烈，考虑平滑预处理或更长时间步聚合")

    if autocorr.get("autocorr_lag1", 0) < 0.5:
        issues.append(f"时序自相关弱 (lag-1 = {autocorr.get('autocorr_lag1', 0):.3f})")
        suggestions.append("时序结构弱，考虑添加外部特征或使用非时序模型作为对比")

    if dist["outlier_ratio_5sigma"] > 1:
        issues.append(f"极端值多 (|z|>5 占 {dist['outlier_ratio_5sigma']:.1f}%)")
        suggestions.append("增大 clip_k 或使用更鲁棒的标准化方法")

    if variance["low_var_features"] > 0:
        issues.append(f"{variance['low_var_features']} 个常数特征")
        suggestions.append("移除常数特征以减少模型复杂度")

    if stationarity["mean_drift"] > 0.5:
        issues.append("数据非平稳 (均值漂移)")
        suggestions.append("考虑使用差分或去趋势处理")

    if problem_features["low_autocorr_features"] > problem_features["high_autocorr_features"] * 2:
        pct = 100 * problem_features["low_autocorr_features"] / x.shape[1]
        issues.append(f"{pct:.0f}% 特征难以时序预测 (autocorr < 0.3)")
        suggestions.append("大量特征时序结构弱，VAE 可能主要学习分布而非时序模式")

    if snr["snr_smooth"] < 5:
        issues.append(f"信噪比低 (SNR = {snr['snr_smooth']:.1f} dB)")
        suggestions.append(f"理论最优 MSE ≈ {snr['theoretical_min_mse']:.3f}，当前模型已接近噪声极限")

    # 打印总结
    if issues:
        print("\n  【发现的问题】")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")

        print("\n  【改进建议】")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"    {i}. {suggestion}")
    else:
        print("  ✓ 数据质量良好，未发现明显问题")

    # 最终结论
    print("\n" + "-" * 60)
    print("【结论】")

    model_mse = cli_args.model_mse
    baseline_mse = persistence["persistence_mse_std"]
    theoretical_min = snr.get("theoretical_min_mse", 0)

    if model_mse <= baseline_mse * 1.1:
        if theoretical_min > 0.5:
            print(f"  模型 MSE ({model_mse:.3f}) 接近 persistence baseline ({baseline_mse:.3f})")
            print(f"  由于数据噪声较大 (理论最优 MSE ≈ {theoretical_min:.3f})，")
            print(f"  当前性能可能已接近该数据集的可预测性极限。")
        else:
            print(f"  模型 MSE ({model_mse:.3f}) 接近 baseline ({baseline_mse:.3f})，但理论上还有提升空间。")
            print(f"  建议检查模型架构或超参数。")
    else:
        print(f"  模型 MSE ({model_mse:.3f}) 高于 baseline ({baseline_mse:.3f})，")
        print(f"  模型可能存在训练问题，建议运行 diagnose_collapse.py 检查。")

    print("-" * 60)


if __name__ == "__main__":
    main()
