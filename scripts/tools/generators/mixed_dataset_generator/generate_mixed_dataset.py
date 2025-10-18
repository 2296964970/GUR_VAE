"""
Mixed dataset generation tool.

This script builds a time-ordered mixed dataset from a clean (normal) CSV
and an attacked CSV with corresponding binary labels. At every timestamp,
the mixed dataset chooses either the clean row (all-one labels) or the
attacked row (labels from the labels CSV). The global time order is kept
by aligning rows by the same time index.

Notes:
- User-facing messages are printed in Chinese for local usability.
- Code, identifiers, and comments are in English per repository guidelines.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def generate_mixed_dataset(
    normal_data_path: str,
    attack_data_path: str,
    attack_labels_path: str,
    output_dir: str,
    attack_ratio: float = 0.3,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a mixed dataset and labels while preserving time order.

    For each timestamp index t in [0, N), choose either the clean row from
    `normal_data_path` (labels are all ones) or the attacked row from
    `attack_data_path` with labels from `attack_labels_path`.

    Args:
        normal_data_path: Path to clean (non-attacked) CSV.
        attack_data_path: Path to attacked CSV (same header and length).
        attack_labels_path: Path to attacked labels CSV (binary, same header/order).
        output_dir: Directory to write `mixed_data.csv` and `mixed_labels.csv`.
        attack_ratio: Fraction of timestamps to take from the attacked CSV.
        random_seed: RNG seed used for selecting attacked positions.

    Returns:
        A tuple of (mixed_data_df, mixed_labels_df).
    """
    np.random.seed(random_seed)

    print("正在读取数据...")
    normal_df = pd.read_csv(normal_data_path)
    attack_df = pd.read_csv(attack_data_path)
    attack_labels_df = pd.read_csv(attack_labels_path)

    print(f"正常数据: {len(normal_df)} 条")
    print(f"攻击数据: {len(attack_df)} 条")

    # Basic schema checks
    if 'timestamp' not in normal_df.columns or 'timestamp' not in attack_df.columns:
        raise ValueError("Missing 'timestamp' column in input CSVs.")
    if list(attack_df.columns) != list(normal_df.columns):
        raise ValueError("Attack CSV header must match normal CSV header exactly.")
    if list(attack_labels_df.columns) != ['timestamp'] + [c for c in normal_df.columns if c != 'timestamp']:
        raise ValueError("Labels CSV header must be ['timestamp'] + feature columns matching the data CSVs.")

    # Determine total length (use min to be robust, but expect equal lengths)
    total_samples = min(len(normal_df), len(attack_df), len(attack_labels_df))
    if not (len(normal_df) == len(attack_df) == len(attack_labels_df) == total_samples):
        print("警告：输入文件长度不一致，将按最短长度对齐。")

    # Compute counts
    n_attack = int(total_samples * attack_ratio)
    n_normal = total_samples - n_attack

    print("\n生成混合数据集:")
    print(f"  总样本数: {total_samples}")
    print(f"  正常样本: {n_normal} ({(1.0 - attack_ratio) * 100:.1f}%)")
    print(f"  攻击样本: {n_attack} ({attack_ratio * 100:.1f}%)")

    # Randomly choose timestamp indices for attacked rows, keep global order
    all_indices = np.arange(total_samples)
    attack_positions = np.sort(np.random.choice(total_samples, n_attack, replace=False))
    attack_positions_set = set(attack_positions)
    print("  保持时间顺序，随机选择时间点作为攻击/正常")

    feature_cols = [c for c in normal_df.columns if c != 'timestamp']

    mixed_data_list = []
    mixed_labels_list = []
    for pos in range(total_samples):
        if pos in attack_positions_set:
            # Take attacked row and its labels at the same timestamp index
            mixed_data_list.append(attack_df.iloc[pos])
            mixed_labels_list.append(attack_labels_df.iloc[pos].to_dict())
        else:
            # Take clean row and create all-ones labels for features
            mixed_data_list.append(normal_df.iloc[pos])
            label_dict = {col: 1 for col in feature_cols}
            label_dict['timestamp'] = normal_df['timestamp'].iloc[pos]
            mixed_labels_list.append(label_dict)

    # Materialize DataFrames
    mixed_data = pd.DataFrame(mixed_data_list).reset_index(drop=True)
    mixed_labels = pd.DataFrame(mixed_labels_list).reset_index(drop=True)

    # Standardize timestamp format to ensure exact match
    for df in (mixed_data, mixed_labels):
        ts = pd.to_datetime(df['timestamp'], errors='coerce')
        df['timestamp'] = ts.dt.strftime('%Y/%m/%d %H:%M')

    # Write outputs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_output_path = output_path / 'mixed_data.csv'
    labels_output_path = output_path / 'mixed_labels.csv'
    mixed_data.to_csv(data_output_path, index=False)
    mixed_labels.to_csv(labels_output_path, index=False)

    print("\n数据集已保存:")
    print(f"  混合数据: {data_output_path}")
    print(f"  标签数据: {labels_output_path}")

    # Label stats
    print("\n标签统计:")
    total_labels = mixed_labels[feature_cols].to_numpy()
    attack_count = int(np.sum(total_labels == 0))
    normal_count = int(np.sum(total_labels == 1))
    total = attack_count + normal_count
    ratio = (attack_count / total * 100.0) if total > 0 else 0.0
    print(f"  标签为0的数量（被攻击）: {attack_count}")
    print(f"  标签为1的数量（正常）: {normal_count}")
    print(f"  攻击特征占比: {ratio:.2f}%")

    # Time order sanity
    print("\n时间顺序验证:")
    print(f"  第一个时间点: {mixed_data['timestamp'].iloc[0]}")
    print(f"  最后一个时间点: {mixed_data['timestamp'].iloc[-1]}")

    return mixed_data, mixed_labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description='生成混合数据集用于评估（保持时间顺序）'
    )
    parser.add_argument(
        '--normal-data', type=str, required=True, help='正常数据csv文件路径'
    )
    parser.add_argument(
        '--attack-data', type=str, required=True, help='攻击数据csv文件路径'
    )
    parser.add_argument(
        '--attack-labels', type=str, required=True, help='攻击标签csv文件路径'
    )
    parser.add_argument(
        '--output-dir', type=str, required=True, help='输出目录'
    )
    parser.add_argument(
        '--attack-ratio', type=float, default=0.3, help='攻击数据占比（0-1之间）'
    )
    parser.add_argument(
        '--random-seed', type=int, default=42, help='随机种子'
    )

    args = parser.parse_args()

    generate_mixed_dataset(
        normal_data_path=args.normal_data,
        attack_data_path=args.attack_data,
        attack_labels_path=args.attack_labels,
        output_dir=args.output_dir,
        attack_ratio=args.attack_ratio,
        random_seed=args.random_seed,
    )


if __name__ == '__main__':
    main()
