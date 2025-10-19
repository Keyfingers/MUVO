"""测试多Town数据加载器"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from muvo.dataset.anovox_dataset import AnoVoxDataset

print("=" * 60)
print("测试多Town数据加载器")
print("=" * 60)

data_root = "/root/autodl-tmp/datasets/AnoVox"

# 测试1: 加载Town07作为训练集的80%
print("\n【测试1】Town07 - 训练集 (80%)")
train_dataset = AnoVoxDataset(
    data_root=data_root,
    split='train',
    towns=['Town07'],
    train_ratio=0.8,
    load_anomaly_labels=True
)
print(f"训练样本数: {len(train_dataset)}")

# 测试2: 加载Town07作为验证集的20%
print("\n【测试2】Town07 - 验证集 (20%)")
val_dataset = AnoVoxDataset(
    data_root=data_root,
    split='val',
    towns=['Town07'],
    train_ratio=0.8,
    load_anomaly_labels=True
)
print(f"验证样本数: {len(val_dataset)}")

# 测试3: 检查标签分布
print("\n【测试3】检查标签分布")
anomaly_count = 0
normal_count = 0

for i in range(min(100, len(train_dataset))):
    sample = train_dataset[i]
    if 'anomaly_label' in sample:
        label_dict = sample['anomaly_label']
        has_anomaly = (label_dict.get('anomaly_is_alive', 'False').lower() == 'true')
        if has_anomaly:
            anomaly_count += 1
        else:
            normal_count += 1

print(f"前100个训练样本:")
print(f"  异常: {anomaly_count} ({100*anomaly_count/(anomaly_count+normal_count):.1f}%)")
print(f"  正常: {normal_count} ({100*normal_count/(anomaly_count+normal_count):.1f}%)")

print("\n✅ 测试完成!")
print("\n注意: Town07全是异常是正常的")
print("一旦下载Town01，标签分布将变为正常!")

