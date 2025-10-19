"""
🔬 验证V3精确映射（基于EV_POSITION锚点）
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from muvo.config import _C
from muvo.dataset.anovox_dataset import AnoVoxDataset, collate_fn
from torch.utils.data import DataLoader
from precise_voxel_mapping_v3 import create_precise_point_labels_v3, visualize_point_labels


def main():
    print("=" * 80)
    print("🔬 验证V3精确映射（基于EV_POSITION锚点）")
    print("=" * 80)
    
    cfg = _C.clone()
    
    print(f"\n📋 配置参数:")
    print(f"   VOXEL.RESOLUTION: {cfg.VOXEL.RESOLUTION}")
    print(f"   VOXEL.SIZE: {cfg.VOXEL.SIZE}")
    print(f"   VOXEL.EV_POSITION: {cfg.VOXEL.EV_POSITION}")
    print(f"\n   推导的网格原点:")
    print(f"   X: 0 - {cfg.VOXEL.EV_POSITION[0]} * {cfg.VOXEL.RESOLUTION} = {-cfg.VOXEL.EV_POSITION[0] * cfg.VOXEL.RESOLUTION}")
    print(f"   Y: 0 - {cfg.VOXEL.EV_POSITION[1]} * {cfg.VOXEL.RESOLUTION} = {-cfg.VOXEL.EV_POSITION[1] * cfg.VOXEL.RESOLUTION}")
    print(f"   Z: 0 - {cfg.VOXEL.EV_POSITION[2]} * {cfg.VOXEL.RESOLUTION} = {-cfg.VOXEL.EV_POSITION[2] * cfg.VOXEL.RESOLUTION}")
    
    # 创建数据集
    dataset = AnoVoxDataset(
        data_root='/root/autodl-tmp/datasets/AnoVox/AnoVox_Dynamic_Mono_Town07',
        split='train',
        load_anomaly_labels=True,
        load_voxel=True
    )
    
    print(f"\n✅ 数据集: {len(dataset)} 样本")
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  设备: {device}")
    
    # 测试前几个批次
    total_samples = 0
    total_points = 0
    total_anomaly_points = 0
    anomaly_samples = 0
    
    visualization_done = False
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 5:  # 测试5个批次
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
        
        # 转换为numpy
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
        
        # 生成V3标签
        labels = create_precise_point_labels_v3(
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
                
                # 可视化第一个有异常的样本
                if not visualization_done:
                    print(f"\n📊 可视化第一个异常样本...")
                    points_i = points[i].cpu().numpy()[:, :3]  # [N, 3]
                    
                    visualize_point_labels(
                        points=points_i,
                        labels=labels_i,
                        save_path='v3_label_visualization.png'
                    )
                    
                    # 打印一些异常点的坐标
                    anomaly_points = points_i[labels_i == 1.0]
                    print(f"\n前10个异常点的坐标:")
                    for j in range(min(10, len(anomaly_points))):
                        px, py, pz = anomaly_points[j]
                        print(f"   点 {j}: ({px:.2f}, {py:.2f}, {pz:.2f})")
                    
                    print(f"\n异常点范围:")
                    print(f"   X: [{anomaly_points[:, 0].min():.2f}, {anomaly_points[:, 0].max():.2f}]")
                    print(f"   Y: [{anomaly_points[:, 1].min():.2f}, {anomaly_points[:, 1].max():.2f}]")
                    print(f"   Z: [{anomaly_points[:, 2].min():.2f}, {anomaly_points[:, 2].max():.2f}]")
                    
                    visualization_done = True
    
    # 总体统计
    print(f"\n{'='*80}")
    print(f"📊 总体统计")
    print(f"{'='*80}")
    print(f"总样本数: {total_samples}")
    print(f"有异常的样本: {anomaly_samples} ({100*anomaly_samples/total_samples if total_samples > 0 else 0:.1f}%)")
    print(f"总点数: {total_points:,}")
    
    if total_points > 0:
        print(f"异常点: {total_anomaly_points:,} ({100*total_anomaly_points/total_points:.2f}%)")
        print(f"正常点: {total_points - total_anomaly_points:,} ({100*(total_points - total_anomaly_points)/total_points:.2f}%)")
        
        if total_anomaly_points > 0:
            pos_weight = (total_points - total_anomaly_points) / total_anomaly_points
            print(f"\n💡 建议的 pos_weight: {pos_weight:.1f}")
            print(f"\n✅✅✅ V3映射验证成功！检测到{total_anomaly_points:,}个异常点！")
            print(f"\n🎯 下一步：集成到训练脚本，开始训练！")
        else:
            print(f"\n⚠️ 警告：仍然没有检测到异常点")
            print(f"可能的原因:")
            print(f"1. 异常语义ID集合需要调整")
            print(f"2. 坐标系仍有问题")
    else:
        print(f"⚠️ 没有处理任何点")


if __name__ == '__main__':
    main()


