"""测试所有样本标签"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from muvo.dataset.anovox_dataset import AnoVoxDataset

data_root = "/root/autodl-tmp/datasets/AnoVox"

dataset = AnoVoxDataset(
    data_root=data_root,
    split='train',
    dataset_types=['Dynamic_Mono_Town07', 'Normality_Mono_Town07'],
    train_ratio=0.8,
    load_anomaly_labels=True
)

print(f"总样本数: {len(dataset)}")

# 统计所有样本
anomaly_count = 0
normal_count = 0
first_normal_idx = -1

for i in range(len(dataset)):
    sample = dataset[i]
    label_dict = sample.get('anomaly_label', {})
    has_anomaly = (label_dict.get('anomaly_is_alive', 'False').lower() == 'true')
    
    if has_anomaly:
        anomaly_count += 1
    else:
        normal_count += 1
        if first_normal_idx == -1:
            first_normal_idx = i

print(f"\n✅ 完整统计:")
print(f"  异常: {anomaly_count} ({100*anomaly_count/(anomaly_count+normal_count):.1f}%)")
print(f"  正常: {normal_count} ({100*normal_count/(anomaly_count+normal_count):.1f}%)")
print(f"  第一个正常样本索引: {first_normal_idx}")

if normal_count > 0:
    print("\n🎉 成功！数据集包含正常和异常场景！")
    
    # 显示几个正常样本
    print("\n前5个正常样本:")
    count = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        label_dict = sample.get('anomaly_label', {})
        has_anomaly = (label_dict.get('anomaly_is_alive', 'False').lower() == 'true')
        if not has_anomaly:
            print(f"  [{i}] {sample['scenario'][:50]}")
            count += 1
            if count >= 5:
                break
else:
    print("\n❌ 失败！")

