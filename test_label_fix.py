"""测试修复后的标签"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from muvo.dataset.anovox_dataset import AnoVoxDataset

data_root = "/root/autodl-tmp/datasets/AnoVox"

# 测试混合数据集
dataset = AnoVoxDataset(
    data_root=data_root,
    split='train',
    dataset_types=['Dynamic_Mono_Town07', 'Normality_Mono_Town07'],
    train_ratio=0.8,
    load_anomaly_labels=True
)

print(f"总样本数: {len(dataset)}")
print("\n前200个样本:")

anomaly_count = 0
normal_count = 0

for i in range(min(200, len(dataset))):
    sample = dataset[i]
    label_dict = sample.get('anomaly_label', {})
    has_anomaly = (label_dict.get('anomaly_is_alive', 'False').lower() == 'true')
    
    if i < 10:
        status = "🔴 异常" if has_anomaly else "🟢 正常"
        print(f"{i:3d}. {sample['scenario'][:40]:40s} {status}")
    
    if has_anomaly:
        anomaly_count += 1
    else:
        normal_count += 1

print(f"\n标签分布:")
print(f"  异常: {anomaly_count} ({100*anomaly_count/(anomaly_count+normal_count):.1f}%)")
print(f"  正常: {normal_count} ({100*normal_count/(anomaly_count+normal_count):.1f}%)")

if normal_count > 0:
    print("\n✅ 成功！数据集包含正常和异常场景！")
else:
    print("\n❌ 失败！仍然只有异常场景！")

