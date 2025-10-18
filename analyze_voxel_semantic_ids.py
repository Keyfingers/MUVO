#!/usr/bin/env python3
"""分析体素文件中的语义ID分布"""
import numpy as np
from pathlib import Path

voxel_file = Path("/root/autodl-tmp/datasets/AnoVox/AnoVox_Dynamic_Mono_Town07/Scenario_54427809-3ef8-4658-bb02-dbd9c74425aa/VOXEL_GRID/VOXEL_GRID_3323.npy")

print("=" * 70)
print(f"📊 分析体素语义ID分布")
print("=" * 70)

data = np.load(voxel_file)
print(f"\n✅ 数据shape: {data.shape}")

semantic_ids = data[:, 3]
unique_ids, counts = np.unique(semantic_ids, return_counts=True)

print(f"\n📋 唯一的语义ID:")
for uid, count in zip(unique_ids, counts):
    print(f"   ID {uid:3d}: {count:6d} 个体素 ({100*count/len(semantic_ids):.2f}%)")

print(f"\n🔍 检查是否有异常ID (CARLA标准: 14-18)")
ANOMALY_IDS = {14, 15, 16, 17, 18}
has_anomaly = any(sid in ANOMALY_IDS for sid in unique_ids)
print(f"   包含异常ID? {has_anomaly}")

if not has_anomaly:
    print("\n⚠️  警告：体素文件中没有任何异常ID！")
    print("   可能原因：")
    print("   1. 这个场景本身没有异常物体")
    print("   2. AnoVox使用的语义ID编码与CARLA标准不同")
    print("   3. 需要查看ANOMALY_*.csv文件来获取真正的异常信息")

# 检查多个文件
print("\n" + "=" * 70)
print("📦 检查多个体素文件...")
print("=" * 70)

voxel_dir = Path("/root/autodl-tmp/datasets/AnoVox/AnoVox_Dynamic_Mono_Town07/Scenario_54427809-3ef8-4658-bb02-dbd9c74425aa/VOXEL_GRID")
voxel_files = sorted(voxel_dir.glob("VOXEL_GRID_*.npy"))[:5]

for vf in voxel_files:
    data = np.load(vf)
    semantic_ids = data[:, 3]
    unique_ids = np.unique(semantic_ids)
    print(f"\n{vf.name}: IDs={list(unique_ids)}")

