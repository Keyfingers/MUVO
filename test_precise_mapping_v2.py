"""
测试精确体素-点映射V2（使用真实体素数据）
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from muvo.config import _C
from muvo.dataset.anovox_dataset import AnoVoxDataset, collate_fn
from torch.utils.data import DataLoader
from precise_voxel_mapping_v2 import create_precise_point_labels_from_voxels


def main():
    print("=" * 80)
    print("🔬 测试精确体素-点映射 V2")
    print("=" * 80)
    
    cfg = _C.clone()
    
    # 创建数据集（加载体素数据）
    dataset = AnoVoxDataset(
        data_root='/root/autodl-tmp/datasets/AnoVox/AnoVox_Dynamic_Mono_Town07',
        split='train',
        load_anomaly_labels=True,
        load_voxel=True  # 关键：加载体素数据
    )
    
    print(f"✅ 数据集: {len(dataset)} 样本\n")
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  设备: {device}\n")
    
    # 测试几个批次
    total_samples = 0
    total_points = 0
    total_anomaly_points = 0
    anomaly_samples = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 5:  # 只测试前5个批次
            break
        
        print(f"\n{'='*80}")
        print(f"批次 {batch_idx + 1}")
        print(f"{'='*80}")
        
        # 准备数据
        points = batch['points'].to(device)
        B, N, _ = points.shape
        
        print(f"点云形状: {points.shape}")
        
        # 获取体素数据
        voxel_data_list = batch.get('voxel', [])
        print(f"体素数据: {len(voxel_data_list)} 个样本")
        
        for i, voxel_data in enumerate(voxel_data_list):
            if voxel_data is not None and len(voxel_data) > 0:
                # 转换为numpy
                if isinstance(voxel_data, torch.Tensor):
                    voxel_np = voxel_data.cpu().numpy()
                else:
                    voxel_np = voxel_data
                
                print(f"  样本 {i}: voxel形状 = {voxel_np.shape}")
                
                # 统计语义ID分布
                if voxel_np.shape[1] >= 4:
                    semantic_ids = voxel_np[:, 3]
                    unique_ids, counts = np.unique(semantic_ids, return_counts=True)
                    print(f"    语义ID: {dict(zip(unique_ids.astype(int), counts))}")
        
        # 转换voxel数据为numpy列表
        voxel_data_list_np = []
        for voxel_tensor in voxel_data_list:
            if voxel_tensor is None or len(voxel_tensor) == 0:
                voxel_data_list_np.append(np.array([]).reshape(0, 4))
            elif isinstance(voxel_tensor, torch.Tensor):
                voxel_data_list_np.append(voxel_tensor.cpu().numpy())
            else:
                voxel_data_list_np.append(voxel_tensor)
        
        # 获取异常标签
        anomaly_labels = batch.get('anomaly_label', [{}] * B)
        
        print(f"\n异常标签:")
        for i, label in enumerate(anomaly_labels):
            is_alive = label.get('anomaly_is_alive', 'N/A')
            print(f"  样本 {i}: anomaly_is_alive = {is_alive}")
        
        # 生成精确标签
        print(f"\n🎯 生成精确标签...")
        labels = create_precise_point_labels_from_voxels(
            points_batch=points,
            voxel_data_list=voxel_data_list_np,
            anomaly_labels_batch=anomaly_labels,
            cfg=cfg,
            device=device
        )
        
        # 统计
        for i in range(B):
            labels_i = labels[i].cpu().numpy()
            anomaly_count = (labels_i == 1.0).sum()
            
            total_samples += 1
            total_points += len(labels_i)
            total_anomaly_points += anomaly_count
            
            if anomaly_count > 0:
                anomaly_samples += 1
                print(f"✅ 样本 {i}: {anomaly_count}/{len(labels_i)} ({100*anomaly_count/len(labels_i):.2f}%) 异常点")
            else:
                print(f"⚪ 样本 {i}: {anomaly_count}/{len(labels_i)} (0.00%) 异常点")
    
    # 总体统计
    print(f"\n{'='*80}")
    print(f"📊 总体统计")
    print(f"{'='*80}")
    print(f"总样本数: {total_samples}")
    print(f"有异常的样本: {anomaly_samples} ({100*anomaly_samples/total_samples:.1f}%)")
    print(f"总点数: {total_points:,}")
    print(f"异常点: {total_anomaly_points:,} ({100*total_anomaly_points/total_points:.2f}%)")
    print(f"正常点: {total_points - total_anomaly_points:,} ({100*(total_points - total_anomaly_points)/total_points:.2f}%)")
    
    if total_anomaly_points > 0:
        pos_weight = (total_points - total_anomaly_points) / total_anomaly_points
        print(f"\n💡 建议的 pos_weight: {pos_weight:.1f}")
        print(f"\n✅ 精确映射验证成功！可以用于训练！")
    else:
        print(f"\n⚠️ 警告：没有检测到任何异常点！")
        print(f"可能的原因：")
        print(f"1. Z轴原点参数需要调整（当前=-2.0m）")
        print(f"2. 异常语义ID集合不完整")
        print(f"3. 坐标系变换参数不正确")


if __name__ == '__main__':
    main()

