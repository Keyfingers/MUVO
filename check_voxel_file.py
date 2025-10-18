#!/usr/bin/env python3
"""快速检查体素文件内容"""
import numpy as np
from pathlib import Path

voxel_file = Path("/root/autodl-tmp/datasets/AnoVox/AnoVox_Dynamic_Mono_Town07/Scenario_54427809-3ef8-4658-bb02-dbd9c74425aa/VOXEL_GRID/VOXEL_GRID_3323.npy")

print("=" * 70)
print(f"📦 检查体素文件: {voxel_file.name}")
print("=" * 70)

data = np.load(voxel_file)

if isinstance(data, np.ndarray):
    print(f"\n✅ 直接numpy数组")
    print(f"   Shape: {data.shape}")
    print(f"   Dtype: {data.dtype}")
    print(f"   前10行:\n{data[:10]}")
else:
    print(f"\n✅ npz文件 (包含多个数组)")
    print(f"   文件keys: {list(data.files)}")
    for key in data.files:
        arr = data[key]
        print(f"\n   Key: '{key}'")
        print(f"      Shape: {arr.shape}")
        print(f"      Dtype: {arr.dtype}")
        print(f"      前10行:\n{arr[:10]}")

