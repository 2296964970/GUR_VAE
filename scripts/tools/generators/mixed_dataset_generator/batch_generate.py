"""
批量生成多个不同攻击比例的混合数据集

该脚本可以一次性生成多个不同攻击比例的数据集，用于全面评估模型性能
"""

from generate_mixed_dataset import generate_mixed_dataset
from pathlib import Path
import argparse


def batch_generate(
    normal_data_path: str,
    attack_data_path: str,
    attack_labels_path: str,
    output_base_dir: str,
    attack_ratios: list = None,
    random_seed: int = 42
):
    """
    批量生成多个不同攻击比例的混合数据集

    Args:
        normal_data_path: 正常数据csv文件路径
        attack_data_path: 攻击数据csv文件路径
        attack_labels_path: 攻击标签csv文件路径
        output_base_dir: 输出基础目录
        attack_ratios: 攻击比例列表，默认为 [0.1, 0.2, 0.3, 0.4, 0.5]
        random_seed: 随机种子
    """
    if attack_ratios is None:
        attack_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

    print("=" * 70)
    print("批量生成混合数据集")
    print("=" * 70)
    print(f"将生成 {len(attack_ratios)} 个不同攻击比例的数据集")
    print(f"攻击比例: {attack_ratios}")
    print("=" * 70)

    base_path = Path(output_base_dir)

    results = []
    for ratio in attack_ratios:
        print(f"\n{'='*70}")
        print(f"正在生成攻击比例为 {ratio*100:.0f}% 的数据集...")
        print(f"{'='*70}")

        # 为每个比例创建单独的目录
        output_dir = base_path / f"attack_ratio_{int(ratio*100):02d}"

        try:
            mixed_data, mixed_labels = generate_mixed_dataset(
                normal_data_path=normal_data_path,
                attack_data_path=attack_data_path,
                attack_labels_path=attack_labels_path,
                output_dir=str(output_dir),
                attack_ratio=ratio,
                random_seed=random_seed
            )

            results.append({
                'ratio': ratio,
                'output_dir': str(output_dir),
                'n_samples': len(mixed_data),
                'success': True
            })

        except Exception as e:
            print(f"错误: 生成攻击比例为 {ratio} 的数据集时失败: {e}")
            results.append({
                'ratio': ratio,
                'output_dir': str(output_dir),
                'success': False,
                'error': str(e)
            })

    # 打印总结
    print(f"\n{'='*70}")
    print("批量生成完成！")
    print(f"{'='*70}")
    print("\n生成结果总结:")
    for result in results:
        if result['success']:
            print(f"  [OK] 攻击比例 {result['ratio']*100:.0f}%: {result['output_dir']}")
            print(f"       样本数: {result['n_samples']}")
        else:
            print(f"  [FAIL] 攻击比例 {result['ratio']*100:.0f}%: 失败")
            print(f"         错误: {result.get('error', 'Unknown')}")

    # 创建一个索引文件
    index_file = base_path / "datasets_index.txt"
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write("混合数据集索引\n")
        f.write("=" * 70 + "\n")
        f.write(f"随机种子: {random_seed}\n")
        f.write("\n数据集列表:\n")
        for result in results:
            if result['success']:
                f.write(f"\n攻击比例: {result['ratio']*100:.0f}%\n")
                f.write(f"  目录: {result['output_dir']}\n")
                f.write(f"  样本数: {result['n_samples']}\n")
                f.write(f"  数据文件: mixed_data.csv\n")
                f.write(f"  标签文件: mixed_labels.csv\n")

    print(f"\n索引文件已保存: {index_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='批量生成多个不同攻击比例的混合数据集')
    parser.add_argument(
        '--normal-data',
        type=str,
        required=True,
        help='正常数据csv文件路径'
    )
    parser.add_argument(
        '--attack-data',
        type=str,
        required=True,
        help='攻击数据csv文件路径'
    )
    parser.add_argument(
        '--attack-labels',
        type=str,
        required=True,
        help='攻击标签csv文件路径'
    )
    parser.add_argument(
        '--output-base-dir',
        type=str,
        required=True,
        help='输出基础目录'
    )
    parser.add_argument(
        '--attack-ratios',
        type=float,
        nargs='+',
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        help='攻击比例列表，例如: 0.1 0.2 0.3'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='随机种子，默认42'
    )

    args = parser.parse_args()

    batch_generate(
        normal_data_path=args.normal_data,
        attack_data_path=args.attack_data,
        attack_labels_path=args.attack_labels,
        output_base_dir=args.output_base_dir,
        attack_ratios=args.attack_ratios,
        random_seed=args.random_seed
    )


if __name__ == '__main__':
    main()
