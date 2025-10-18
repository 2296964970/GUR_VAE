"""验证生成的混合数据集"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def verify_mixed_dataset(data_path: str, labels_path: str):
    """
    验证混合数据集的质量和完整性

    Args:
        data_path: 混合数据文件路径
        labels_path: 标签文件路径
    """
    # 读取生成的数据
    data = pd.read_csv(data_path)
    labels = pd.read_csv(labels_path)

    print("=" * 60)
    print("混合数据集验证")
    print("=" * 60)

    print(f"\n数据形状: {data.shape}")
    print(f"标签形状: {labels.shape}")

    print(f"\n数据前3行:")
    print(data.head(3))

    print(f"\n标签前3行:")
    print(labels.head(3))

    # 检查每行标签
    print(f"\n标签统计（前10行）:")
    for i in range(min(10, len(labels))):
        label_row = labels.iloc[i, 1:].values
        n_zeros = sum(label_row == 0)
        n_ones = sum(label_row == 1)
        is_all_ones = all(label_row == 1)

        if is_all_ones:
            print(f"  第{i}行: 全部为1 (正常样本)")
        else:
            print(f"  第{i}行: 0的数量={n_zeros}, 1的数量={n_ones} (攻击样本)")

    # 统计有多少行是全1的
    all_ones_count = 0
    attack_count = 0
    for i in range(len(labels)):
        label_row = labels.iloc[i, 1:].values
        if all(label_row == 1):
            all_ones_count += 1
        else:
            attack_count += 1

    print(f"\n总体统计:")
    print(f"  正常样本（标签全为1）: {all_ones_count} ({all_ones_count/len(labels)*100:.1f}%)")
    print(f"  攻击样本（标签有0）: {attack_count} ({attack_count/len(labels)*100:.1f}%)")

    # 验证数据和标签的timestamp是否对应
    print(f"\n验证数据完整性:")
    timestamp_match = all(data['timestamp'] == labels['timestamp'])
    print(f"  数据和标签的timestamp是否匹配: {timestamp_match}")

    # 检查是否有缺失值
    print(f"\n缺失值检查:")
    print(f"  数据中的缺失值: {data.isnull().sum().sum()}")
    print(f"  标签中的缺失值: {labels.isnull().sum().sum()}")

    # 检查时间顺序
    print(f"\n时间顺序检查:")
    print(f"  第一个时间点: {data['timestamp'].iloc[0]}")
    print(f"  最后一个时间点: {data['timestamp'].iloc[-1]}")

    print("\n验证完成！")

    return {
        'data_shape': data.shape,
        'labels_shape': labels.shape,
        'normal_count': all_ones_count,
        'attack_count': attack_count,
        'timestamp_match': timestamp_match,
        'data_nulls': data.isnull().sum().sum(),
        'labels_nulls': labels.isnull().sum().sum()
    }


def main():
    parser = argparse.ArgumentParser(description='验证混合数据集')
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='混合数据文件路径'
    )
    parser.add_argument(
        '--labels',
        type=str,
        required=True,
        help='标签文件路径'
    )

    args = parser.parse_args()

    verify_mixed_dataset(args.data, args.labels)


if __name__ == '__main__':
    main()
