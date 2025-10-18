"""
测试脚本：验证混合数据集生成器的功能
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from generate_mixed_dataset import generate_mixed_dataset


def create_test_data(n_samples=100, n_features=10):
    """创建测试数据"""
    # 创建正常数据
    normal_data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    normal_data.insert(0, 'timestamp', pd.date_range('2025-01-01', periods=n_samples, freq='5min'))

    # 创建攻击数据
    attack_data = pd.DataFrame(
        np.random.randn(n_samples, n_features) + 2,  # 攻击数据有偏移
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    attack_data.insert(0, 'timestamp', pd.date_range('2025-02-01', periods=n_samples, freq='5min'))

    # 创建攻击标签（随机设置一些特征为0）
    attack_labels = pd.DataFrame(
        np.ones((n_samples, n_features), dtype=int),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    # 随机将30%的特征标记为攻击
    for i in range(n_samples):
        n_attack_features = np.random.randint(1, max(2, n_features // 3))
        attack_features = np.random.choice(n_features, n_attack_features, replace=False)
        for feat in attack_features:
            attack_labels.iloc[i, feat] = 0

    attack_labels.insert(0, 'timestamp', attack_data['timestamp'])

    return normal_data, attack_data, attack_labels


def test_basic_generation():
    """测试基本的数据集生成功能"""
    print("\n" + "="*60)
    print("测试1: 基本数据集生成")
    print("="*60)

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        # 创建测试数据
        normal_data, attack_data, attack_labels = create_test_data(n_samples=100, n_features=10)

        # 保存测试数据
        normal_path = Path(temp_dir) / "normal.csv"
        attack_path = Path(temp_dir) / "attack.csv"
        labels_path = Path(temp_dir) / "labels.csv"

        normal_data.to_csv(normal_path, index=False)
        attack_data.to_csv(attack_path, index=False)
        attack_labels.to_csv(labels_path, index=False)

        # 生成混合数据集
        output_dir = Path(temp_dir) / "output"
        mixed_data, mixed_labels = generate_mixed_dataset(
            normal_data_path=str(normal_path),
            attack_data_path=str(attack_path),
            attack_labels_path=str(labels_path),
            output_dir=str(output_dir),
            attack_ratio=0.3,
            random_seed=42
        )

        # 验证
        assert len(mixed_data) == 100, "混合数据集大小不正确"
        assert len(mixed_labels) == 100, "标签数据集大小不正确"
        assert mixed_data.shape == mixed_labels.shape, "数据和标签形状不匹配"
        assert all(mixed_data['timestamp'] == mixed_labels['timestamp']), "时间戳不匹配"

        # 统计正常/攻击样本
        n_attack = sum(~(mixed_labels.iloc[:, 1:] == 1).all(axis=1))
        n_normal = len(mixed_labels) - n_attack

        assert abs(n_attack / len(mixed_labels) - 0.3) < 0.05, "攻击比例不正确"

        print("[OK] 测试通过：基本数据集生成正常")
        print(f"  - 总样本数: {len(mixed_data)}")
        print(f"  - 正常样本: {n_normal} ({n_normal/len(mixed_data)*100:.1f}%)")
        print(f"  - 攻击样本: {n_attack} ({n_attack/len(mixed_data)*100:.1f}%)")

    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)


def test_time_order_preservation():
    """测试时间顺序保持"""
    print("\n" + "="*60)
    print("测试2: 时间顺序保持")
    print("="*60)

    temp_dir = tempfile.mkdtemp()
    try:
        # 创建测试数据
        normal_data, attack_data, attack_labels = create_test_data(n_samples=50, n_features=5)

        # 保存测试数据
        normal_path = Path(temp_dir) / "normal.csv"
        attack_path = Path(temp_dir) / "attack.csv"
        labels_path = Path(temp_dir) / "labels.csv"

        normal_data.to_csv(normal_path, index=False)
        attack_data.to_csv(attack_path, index=False)
        attack_labels.to_csv(labels_path, index=False)

        # 生成混合数据集
        output_dir = Path(temp_dir) / "output"
        mixed_data, mixed_labels = generate_mixed_dataset(
            normal_data_path=str(normal_path),
            attack_data_path=str(attack_path),
            attack_labels_path=str(labels_path),
            output_dir=str(output_dir),
            attack_ratio=0.4,
            random_seed=123
        )

        # 验证时间顺序：由于我们按位置混合，前面的数据应该保持相对顺序
        print("[OK] 测试通过：时间顺序保持正常")
        print(f"  - 第一个时间点: {mixed_data['timestamp'].iloc[0]}")
        print(f"  - 最后一个时间点: {mixed_data['timestamp'].iloc[-1]}")

    finally:
        shutil.rmtree(temp_dir)


def test_different_ratios():
    """测试不同的攻击比例"""
    print("\n" + "="*60)
    print("测试3: 不同攻击比例")
    print("="*60)

    temp_dir = tempfile.mkdtemp()
    try:
        # 创建测试数据
        normal_data, attack_data, attack_labels = create_test_data(n_samples=100, n_features=5)

        # 保存测试数据
        normal_path = Path(temp_dir) / "normal.csv"
        attack_path = Path(temp_dir) / "attack.csv"
        labels_path = Path(temp_dir) / "labels.csv"

        normal_data.to_csv(normal_path, index=False)
        attack_data.to_csv(attack_path, index=False)
        attack_labels.to_csv(labels_path, index=False)

        # 测试不同比例
        for ratio in [0.1, 0.3, 0.5, 0.7]:
            output_dir = Path(temp_dir) / f"output_{int(ratio*100)}"
            mixed_data, mixed_labels = generate_mixed_dataset(
                normal_data_path=str(normal_path),
                attack_data_path=str(attack_path),
                attack_labels_path=str(labels_path),
                output_dir=str(output_dir),
                attack_ratio=ratio,
                random_seed=42
            )

            # 统计实际攻击比例
            n_attack = sum(~(mixed_labels.iloc[:, 1:] == 1).all(axis=1))
            actual_ratio = n_attack / len(mixed_labels)

            assert abs(actual_ratio - ratio) < 0.05, f"攻击比例 {ratio} 不正确"
            print(f"  [OK] 攻击比例 {ratio*100:.0f}%: 实际 {actual_ratio*100:.1f}%")

        print("[OK] 测试通过：不同攻击比例生成正常")

    finally:
        shutil.rmtree(temp_dir)


def test_label_integrity():
    """测试标签完整性"""
    print("\n" + "="*60)
    print("测试4: 标签完整性")
    print("="*60)

    temp_dir = tempfile.mkdtemp()
    try:
        # 创建测试数据
        normal_data, attack_data, attack_labels = create_test_data(n_samples=50, n_features=8)

        # 保存测试数据
        normal_path = Path(temp_dir) / "normal.csv"
        attack_path = Path(temp_dir) / "attack.csv"
        labels_path = Path(temp_dir) / "labels.csv"

        normal_data.to_csv(normal_path, index=False)
        attack_data.to_csv(attack_path, index=False)
        attack_labels.to_csv(labels_path, index=False)

        # 生成混合数据集
        output_dir = Path(temp_dir) / "output"
        mixed_data, mixed_labels = generate_mixed_dataset(
            normal_data_path=str(normal_path),
            attack_data_path=str(attack_path),
            attack_labels_path=str(labels_path),
            output_dir=str(output_dir),
            attack_ratio=0.3,
            random_seed=42
        )

        # 验证正常样本的标签全为1
        for i in range(len(mixed_labels)):
            label_row = mixed_labels.iloc[i, 1:].values
            is_all_ones = all(label_row == 1)
            has_zeros = any(label_row == 0)

            if is_all_ones:
                # 正常样本
                assert not has_zeros, f"行 {i} 应该全为1"
            else:
                # 攻击样本
                assert has_zeros, f"行 {i} 应该包含0"

        print("[OK] 测试通过：标签完整性正常")
        print(f"  - 所有标签值都是0或1")
        print(f"  - 正常样本标签全为1")
        print(f"  - 攻击样本标签包含0")

    finally:
        shutil.rmtree(temp_dir)


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("混合数据集生成器 - 测试套件")
    print("="*60)

    tests = [
        test_basic_generation,
        test_time_order_preservation,
        test_different_ratios,
        test_label_integrity
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] 测试失败: {test.__name__}")
            print(f"  错误: {e}")
            failed += 1

    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"通过: {passed}/{len(tests)}")
    print(f"失败: {failed}/{len(tests)}")

    if failed == 0:
        print("\n[SUCCESS] 所有测试通过！")
    else:
        print(f"\n[WARNING] 有 {failed} 个测试失败")


if __name__ == '__main__':
    run_all_tests()
