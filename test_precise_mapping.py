#!/usr/bin/env python3
"""测试精确映射函数"""
import sys
import torch
import numpy as np

sys.path.insert(0, '/root/autodl-tmp/MUVO/MUVO')

from muvo.dataset.anovox_dataset import AnoVoxDataset, collate_fn
from precise_label_mapping import create_precise_point_labels
from torch.utils.data import DataLoader

def main():
    print("=" * 70)
    print("🔍 测试精确映射函数")
    print("=" * 70)
    
    # 加载数据集
    dataset = AnoVoxDataset(
        data_root='/root/autodl-tmp/datasets/AnoVox/AnoVox_Dynamic_Mono_Town07',
        split='train',
        load_voxel=True,
        load_anomaly_labels=True
    )
    
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    # 获取一个batch
    batch = next(iter(dataloader))
    
    print(f"\n✅ Batch加载成功")
    print(f"   Image shape: {batch['image'].shape}")
    print(f"   Points shape: {batch['points'].shape}")
    print(f"   Voxel存在: {'voxel' in batch}")
    if 'voxel' in batch:
        print(f"   Voxel是列表，长度: {len(batch['voxel'])}")
    print(f"   Anomaly_label存在: {'anomaly_label' in batch}")
    
    # 准备数据
    B = batch['image'].shape[0]
    points = batch['points']  # [B, N, 4]
    N = points.shape[1]
    
    # 提取体素数据
    voxel_data_list = []
    for i in range(B):
        if 'voxel' in batch:
            voxel = batch['voxel']
            voxel_np = voxel[i].numpy() if isinstance(voxel, torch.Tensor) else voxel[i]
            voxel_data_list.append(voxel_np)
            
            print(f"\n📦 样本 {i}:")
            print(f"   Voxel shape: {voxel_np.shape}")
            print(f"   语义ID唯一值: {np.unique(voxel_np[:, 3])}")
            
            # 检查异常ID
            ANOMALY_IDS = {14, 15, 16, 17, 18}
            anomaly_voxels = sum(1 for sid in voxel_np[:, 3] if sid in ANOMALY_IDS)
            print(f"   异常体素数量: {anomaly_voxels} / {len(voxel_np)} ({100*anomaly_voxels/len(voxel_np):.2f}%)")
        else:
            voxel_data_list.append(np.array([]))
    
    # 调用精确映射
    print(f"\n🎯 调用create_precise_point_labels...")
    print(f"   Points shape: {points.shape}")
    print(f"   Voxel data list length: {len(voxel_data_list)}")
    
    labels = create_precise_point_labels(
        points=points,
        anomaly_labels=batch.get('anomaly_label', [None] * B),
        voxel_data_list=voxel_data_list,
        voxel_resolution=0.2,
        grid_origin=(0.0, 0.0, 0.0),
        anomaly_semantic_id=14,
        device=torch.device('cpu')
    )
    
    print(f"\n✅ 标签生成成功!")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Label values unique: {torch.unique(labels)}")
    print(f"   Anomaly points: {(labels == 1).sum().item()} / {labels.numel()}")
    print(f"   Anomaly ratio: {100*(labels == 1).sum().item()/labels.numel():.2f}%")
    
    for i in range(B):
        sample_labels = labels[i]
        anomaly_count = (sample_labels == 1).sum().item()
        print(f"\n   样本 {i}: {anomaly_count} / {len(sample_labels)} 异常点 ({100*anomaly_count/len(sample_labels):.2f}%)")

if __name__ == '__main__':
    main()

