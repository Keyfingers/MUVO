"""
🔬 验证精确体素-点映射的正确性
"""

import sys
import torch
import numpy as np
from pathlib import Path

# 添加muvo到路径
sys.path.insert(0, str(Path(__file__).parent))

from muvo.config import get_parser, get_cfg
from muvo.dataset.anovox_dataset import AnoVoxDataset, collate_fn
from precise_voxel_mapping import create_precise_point_labels_from_config, verify_coordinate_mapping


def main():
    print("=" * 80)
    print("🔬 精确体素-点映射验证")
    print("=" * 80)
    
    # 加载配置（使用默认配置）
    from muvo.config import _C
    cfg = _C.clone()
    
    # 创建数据集
    dataset = AnoVoxDataset(
        data_root='/root/autodl-tmp/datasets/AnoVox/AnoVox_Dynamic_Mono_Town07',
        split='train',
        load_anomaly_labels=True
    )
    
    print(f"\n✅ 数据集加载成功: {len(dataset)} 个样本")
    
    # 寻找一个有异常的样本
    print("\n🔍 寻找有异常的样本...")
    anomaly_sample_idx = None
    
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        label = sample.get('anomaly_label', {})
        
        if isinstance(label, dict):
            anomaly_is_alive = label.get('anomaly_is_alive', 'false')
            anomaly_coords = label.get('anomaly_coords', [])
            
            if anomaly_is_alive == 'true' and len(anomaly_coords) > 0:
                anomaly_sample_idx = i
                print(f"✅ 找到异常样本: 索引 {i}, 异常体素数量: {len(anomaly_coords)}")
                break
    
    if anomaly_sample_idx is None:
        print("❌ 未找到有异常标注的样本！")
        return
    
    # 获取样本
    sample = dataset[anomaly_sample_idx]
    
    # 打印样本信息
    print(f"\n📦 样本 {anomaly_sample_idx} 信息:")
    print(f"   点云形状: {sample['points'].shape}")
    print(f"   图像形状: {sample['image'].shape}")
    
    label = sample['anomaly_label']
    print(f"   异常状态: {label.get('anomaly_is_alive', 'N/A')}")
    print(f"   异常体素数量: {len(label.get('anomaly_coords', []))}")
    
    # 先进行坐标映射验证
    points_np = sample['points'].numpy()
    anomaly_coords = label.get('anomaly_coords', [])
    
    verify_coordinate_mapping(points_np, anomaly_coords, cfg)
    
    # 测试标签生成
    print("\n" + "=" * 80)
    print("🎯 测试标签生成")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建batch（批次大小=1）
    batch = {
        'points': sample['points'].unsqueeze(0).to(device),  # [1, N, 4]
        'image': sample['image'].unsqueeze(0).to(device),
        'anomaly_label': [label]
    }
    
    # 生成标签
    print("\n🏷️ 生成精确标签...")
    labels = create_precise_point_labels_from_config(
        points_batch=batch['points'],
        anomaly_labels_batch=batch['anomaly_label'],
        cfg=cfg,
        device=device
    )
    
    # 统计结果
    labels_np = labels[0].cpu().numpy()
    anomaly_mask = labels_np == 1.0
    normal_mask = labels_np == 0.0
    
    print(f"\n📊 标签生成结果:")
    print(f"   总点数: {len(labels_np)}")
    print(f"   异常点: {anomaly_mask.sum()} ({100*anomaly_mask.sum()/len(labels_np):.2f}%)")
    print(f"   正常点: {normal_mask.sum()} ({100*normal_mask.sum()/len(labels_np):.2f}%)")
    
    if anomaly_mask.sum() > 0:
        print(f"\n✅ 成功！映射生成了 {anomaly_mask.sum()} 个异常点标签")
        
        # 显示一些异常点的坐标
        anomaly_points = points_np[anomaly_mask]
        print(f"\n前10个异常点的坐标:")
        for i in range(min(10, len(anomaly_points))):
            px, py, pz = anomaly_points[i][:3]
            print(f"   点 {i}: ({px:.2f}, {py:.2f}, {pz:.2f})")
    else:
        print(f"\n⚠️ 警告：没有生成任何异常点标签！")
        print(f"可能的原因:")
        print(f"1. 坐标系原点参数不正确")
        print(f"2. 点云坐标和体素坐标系不匹配")
        print(f"3. Z轴原点假设(-2.0m)不正确")
    
    # 测试多个样本
    print("\n" + "=" * 80)
    print("🧪 批量测试（前10个异常样本）")
    print("=" * 80)
    
    anomaly_samples = []
    for i in range(min(200, len(dataset))):
        sample = dataset[i]
        label = sample.get('anomaly_label', {})
        
        if isinstance(label, dict):
            anomaly_is_alive = label.get('anomaly_is_alive', 'false')
            anomaly_coords = label.get('anomaly_coords', [])
            
            if anomaly_is_alive == 'true' and len(anomaly_coords) > 0:
                anomaly_samples.append((i, sample))
                if len(anomaly_samples) >= 10:
                    break
    
    print(f"\n找到 {len(anomaly_samples)} 个异常样本")
    
    total_points = 0
    total_anomaly_points = 0
    
    for idx, (sample_idx, sample) in enumerate(anomaly_samples):
        points = sample['points'].unsqueeze(0).to(device)
        label = sample['anomaly_label']
        
        labels = create_precise_point_labels_from_config(
            points_batch=points,
            anomaly_labels_batch=[label],
            cfg=cfg,
            device=device
        )
        
        labels_np = labels[0].cpu().numpy()
        anomaly_count = (labels_np == 1.0).sum()
        
        total_points += len(labels_np)
        total_anomaly_points += anomaly_count
        
        print(f"   样本 {sample_idx}: {anomaly_count}/{len(labels_np)} ({100*anomaly_count/len(labels_np):.2f}%) 异常点")
    
    print(f"\n📊 总体统计:")
    print(f"   总点数: {total_points:,}")
    print(f"   异常点: {total_anomaly_points:,} ({100*total_anomaly_points/total_points:.2f}%)")
    print(f"   正常点: {total_points - total_anomaly_points:,} ({100*(total_points - total_anomaly_points)/total_points:.2f}%)")
    
    if total_anomaly_points > 0:
        print(f"\n✅ 验证成功！精确映射可以正常工作！")
        print(f"建议的 pos_weight 值: {(total_points - total_anomaly_points) / total_anomaly_points:.1f}")
    else:
        print(f"\n❌ 验证失败！需要调整参数！")


if __name__ == '__main__':
    main()

