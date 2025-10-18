#!/usr/bin/env python3
"""调试点云到体素的坐标映射"""
import sys
import numpy as np
import torch

sys.path.insert(0, '/root/autodl-tmp/MUVO/MUVO')
from muvo.dataset.anovox_dataset import AnoVoxDataset

def main():
    dataset = AnoVoxDataset(
        data_root='/root/autodl-tmp/datasets/AnoVox/AnoVox_Dynamic_Mono_Town07',
        split='train',
        load_voxel=True
    )
    
    sample = dataset[0]
    points = sample['points'].numpy()  # [N, 4]
    voxel = sample['voxel'].numpy()  # [M, 4]
    
    print("=" * 70)
    print("📊 点云与体素坐标对比")
    print("=" * 70)
    
    print(f"\n点云 (真实世界坐标):")
    print(f"  数量: {len(points)}")
    print(f"  X范围: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Y范围: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  Z范围: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    print(f"\n体素 (体素索引):")
    print(f"  数量: {len(voxel)}")
    print(f"  VX范围: [{voxel[:, 0].min()}, {voxel[:, 0].max()}]")
    print(f"  VY范围: [{voxel[:, 1].min()}, {voxel[:, 1].max()}]")
    print(f"  VZ范围: [{voxel[:, 2].min()}, {voxel[:, 2].max()}]")
    print(f"  语义ID: {np.unique(voxel[:, 3])}")
    
    print(f"\n🔍 尝试推断映射关系:")
    
    # 假设：voxel_idx = floor((point_coord - origin) / resolution)
    # 尝试不同的分辨率
    resolutions = [0.1, 0.2, 0.25, 0.5]
    
    for res in resolutions:
        # 尝试不同的原点
        origins = [
            (0, 0, 0),
            (points[:, 0].min(), points[:, 1].min(), points[:, 2].min()),
            (-50, -50, 0),
        ]
        
        for origin in origins:
            voxel_indices = np.floor((points[:, :3] - np.array(origin)) / res).astype(np.int32)
            
            # 检查范围是否匹配
            vx_min, vx_max = voxel_indices[:, 0].min(), voxel_indices[:, 0].max()
            vy_min, vy_max = voxel_indices[:, 1].min(), voxel_indices[:, 1].max()
            vz_min, vz_max = voxel_indices[:, 2].min(), voxel_indices[:, 2].max()
            
            if (abs(vx_min - voxel[:, 0].min()) < 10 and 
                abs(vx_max - voxel[:, 0].max()) < 10 and
                abs(vy_min - voxel[:, 1].min()) < 10 and
                abs(vy_max - voxel[:, 1].max()) < 10):
                print(f"\n  ✅ 可能的匹配:")
                print(f"     分辨率: {res}")
                print(f"     原点: {origin}")
                print(f"     点云→体素索引范围:")
                print(f"       VX: [{vx_min}, {vx_max}] vs 真实 [{voxel[:, 0].min()}, {voxel[:, 0].max()}]")
                print(f"       VY: [{vy_min}, {vy_max}] vs 真实 [{voxel[:, 1].min()}, {voxel[:, 1].max()}]")
                print(f"       VZ: [{vz_min}, {vz_max}] vs 真实 [{voxel[:, 2].min()}, {voxel[:, 2].max()}]")

if __name__ == '__main__':
    main()

