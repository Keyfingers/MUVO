"""测试混合数据集加载（正常+异常）"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from muvo.dataset.anovox_dataset import AnoVoxDataset

print("=" * 70)
print("测试AnoVox混合数据集加载（Dynamic + Normality）")
print("=" * 70)

data_root = "/root/autodl-tmp/datasets/AnoVox"

# 测试1: 只用Dynamic（当前状态）
print("\n【测试1】只使用Dynamic_Mono_Town07（当前）")
train_dynamic_only = AnoVoxDataset(
    data_root=data_root,
    split='train',
    dataset_types=['Dynamic_Mono_Town07'],
    train_ratio=0.8,
    load_anomaly_labels=True
)
print(f"训练样本数: {len(train_dynamic_only)}")

# 统计标签
anomaly_count = 0
normal_count = 0
for i in range(min(100, len(train_dynamic_only))):
    sample = train_dynamic_only[i]
    if 'anomaly_label' in sample:
        has_anomaly = (sample['anomaly_label'].get('anomaly_is_alive', 'False').lower() == 'true')
        if has_anomaly:
            anomaly_count += 1
        else:
            normal_count += 1

print(f"前100个样本标签分布:")
print(f"  异常: {anomaly_count} ({100*anomaly_count/(anomaly_count+normal_count):.1f}%)")
print(f"  正常: {normal_count} ({100*normal_count/(anomaly_count+normal_count):.1f}%)")

# 测试2: 混合Dynamic + Normality（下载后）
print("\n【测试2】混合Dynamic + Normality（需要先下载Normality）")
try:
    train_mixed = AnoVoxDataset(
        data_root=data_root,
        split='train',
        dataset_types=['Dynamic_Mono_Town07', 'Normality_Mono_Town07'],
        train_ratio=0.8,
        load_anomaly_labels=True
    )
    print(f"训练样本数: {len(train_mixed)}")
    
    # 统计标签
    anomaly_count = 0
    normal_count = 0
    for i in range(min(200, len(train_mixed))):
        sample = train_mixed[i]
        if 'anomaly_label' in sample:
            has_anomaly = (sample['anomaly_label'].get('anomaly_is_alive', 'False').lower() == 'true')
            if has_anomaly:
                anomaly_count += 1
            else:
                normal_count += 1
    
    print(f"前200个样本标签分布:")
    print(f"  异常: {anomaly_count} ({100*anomaly_count/(anomaly_count+normal_count):.1f}%)")
    print(f"  正常: {normal_count} ({100*normal_count/(anomaly_count+normal_count):.1f}%)")
    
    if normal_count > 0:
        print("\n✅ 成功！数据集包含正常和异常场景！")
        print("🎯 可以开始正确的训练了！")
    else:
        print("\n⚠️  仍然只有异常场景，可能Normality数据还没下载")
        
except Exception as e:
    print(f"⚠️  无法加载混合数据集: {e}")
    print("可能原因: Normality_Mono_Town07还未下载")

print("\n" + "=" * 70)
print("提示:")
print("1. 当前只有Dynamic（100%异常）")
print("2. 下载Normality后，将有正常+异常混合数据")
print("3. 下载命令: bash /root/autodl-tmp/datasets/AnoVox/download_normality_town07.sh")
print("=" * 70)

