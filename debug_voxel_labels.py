#!/usr/bin/env python3
"""
调试脚本：检查AnoVox体素标签的实际格式
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/root/autodl-tmp/MUVO/MUVO')

from muvo.dataset.anovox_dataset import AnoVoxDataset

def main():
    print("=" * 70)
    print("🔍 AnoVox体素标签格式调试")
    print("=" * 70)
    
    # 加载数据集
    dataset = AnoVoxDataset(
        data_root='/root/autodl-tmp/datasets/AnoVox/AnoVox_Dynamic_Mono_Town07',
        split='train',
        load_voxel=True,
        load_anomaly_labels=True
    )
    
    print(f"\n✅ 数据集加载成功: {len(dataset)} 样本")
    
    # 检查前5个样本的体素标签
    for idx in range(min(5, len(dataset))):
        print(f"\n{'='*70}")
        print(f"📦 样本 #{idx}")
        print(f"{'='*70}")
        
        sample = dataset[idx]
        
        # 检查voxel_label
        if sample['voxel_label'] is not None:
            voxel_label = sample['voxel_label']
            print(f"\n✅ voxel_label 存在")
            print(f"   类型: {type(voxel_label)}")
            print(f"   Shape: {voxel_label.shape if isinstance(voxel_label, np.ndarray) else 'N/A'}")
            
            if isinstance(voxel_label, np.ndarray):
                print(f"   Dtype: {voxel_label.dtype}")
                print(f"   前5行:")
                print(f"   {voxel_label[:5]}")
                
                # 检查列的含义
                if voxel_label.shape[1] >= 4:
                    print(f"\n   📊 第4列（语义ID）统计:")
                    semantic_ids = voxel_label[:, 3]
                    unique_ids, counts = np.unique(semantic_ids, return_counts=True)
                    for uid, count in zip(unique_ids, counts):
                        print(f"      ID {int(uid):3d}: {count:6d} 个体素")
                    
                    # 检查是否有14号ID（vehicle）
                    ANOMALY_IDS = {14, 15, 16, 17, 18}
                    has_anomaly = any(sid in ANOMALY_IDS for sid in unique_ids)
                    print(f"\n   🚨 是否包含异常ID {ANOMALY_IDS}? {has_anomaly}")
                    if has_anomaly:
                        anomaly_counts = sum(counts[unique_ids == aid][0] for aid in ANOMALY_IDS if aid in unique_ids)
                        print(f"      异常体素数量: {anomaly_counts}")
        else:
            print(f"❌ voxel_label 为 None")
        
        # 检查anomaly_label
        if sample['anomaly_label'] is not None:
            anomaly_label = sample['anomaly_label']
            print(f"\n✅ anomaly_label 存在")
            print(f"   类型: {type(anomaly_label)}")
            if isinstance(anomaly_label, dict):
                print(f"   键: {list(anomaly_label.keys())}")
                print(f"   内容:")
                for k, v in anomaly_label.items():
                    print(f"      {k}: {v}")
        else:
            print(f"❌ anomaly_label 为 None")
        
        # 检查点云
        points = sample['points']
        print(f"\n✅ 点云")
        print(f"   Shape: {points.shape}")
        print(f"   坐标范围:")
        print(f"      X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"      Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"      Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    print(f"\n{'='*70}")
    print("✅ 调试完成")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()

