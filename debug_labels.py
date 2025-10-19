"""调试标签分布"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from muvo.dataset.anovox_dataset import AnoVoxDataset
from muvo.config import _C

# 直接指定数据集路径
data_root = "/root/autodl-tmp/datasets/AnoVox/AnoVox_Dynamic_Mono_Town07"

# 创建数据集
dataset = AnoVoxDataset(
    data_root=data_root,
    split='train',
    load_anomaly_labels=True
)

print(f"总样本数: {len(dataset)}")
print("\n前50个样本的标签:")

anomaly_count = 0
normal_count = 0

for i in range(min(50, len(dataset))):
    sample = dataset[i]
    if 'anomaly_label' in sample:
        label_dict = sample['anomaly_label']
        anomaly_is_alive = label_dict.get('anomaly_is_alive', 'False')
        has_anomaly = (str(anomaly_is_alive).lower() == 'true')
        
        status = "🔴 异常" if has_anomaly else "🟢 正常"
        print(f"{i:3d}. {sample['scenario'][:30]:30s} frame={sample['frame_id']:4s} {status}")
        
        if has_anomaly:
            anomaly_count += 1
        else:
            normal_count += 1
    else:
        print(f"{i:3d}. {sample['scenario'][:30]:30s} frame={sample['frame_id']:4s} ⚠️ 无标签")

print(f"\n统计:")
print(f"  异常: {anomaly_count} ({100*anomaly_count/(anomaly_count+normal_count):.1f}%)")
print(f"  正常: {normal_count} ({100*normal_count/(anomaly_count+normal_count):.1f}%)")

