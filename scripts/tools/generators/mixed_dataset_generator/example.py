"""
示例：如何使用工具生成混合数据集
"""

from generate_mixed_dataset import generate_mixed_dataset
from verify_dataset import verify_mixed_dataset
from pathlib import Path

# 定义数据路径
# 请根据你的实际情况修改这些路径
data_dir = Path(__file__).parent.parent.parent / "input" / "case118"

normal_data_path = data_dir / "case118_acopf_all_rows_noisy.csv"
attack_data_path = data_dir / "case118_fdia_2025-07_2025-08_noisy.csv"
attack_labels_path = data_dir / "case118_fdia_2025-07_2025-08_labels.csv"

# 定义输出目录
output_dir = data_dir / "mixed_eval_example"

print("="*70)
print("示例：生成混合数据集")
print("="*70)

# 生成混合数据集
# attack_ratio=0.3 表示30%的数据是攻击数据，70%是正常数据
mixed_data, mixed_labels = generate_mixed_dataset(
    normal_data_path=str(normal_data_path),
    attack_data_path=str(attack_data_path),
    attack_labels_path=str(attack_labels_path),
    output_dir=str(output_dir),
    attack_ratio=0.3,
    random_seed=42
)

print("\n" + "="*70)
print("混合数据集生成完成！")
print("="*70)
print(f"数据形状: {mixed_data.shape}")
print(f"标签形状: {mixed_labels.shape}")

# 验证生成的数据集
print("\n" + "="*70)
print("开始验证数据集...")
print("="*70)

verify_mixed_dataset(
    data_path=str(output_dir / "mixed_data.csv"),
    labels_path=str(output_dir / "mixed_labels.csv")
)
