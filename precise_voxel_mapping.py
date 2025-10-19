"""
🎯 精确的体素-点标签映射（方案A完整实现）

根据config.py中的参数推导出的精确坐标系变换：
- VOXEL.SIZE = [192, 192, 64]
- VOXEL.RESOLUTION = 0.2
- BEV.OFFSET_FORWARD = -64

推导出的体素网格原点：
- X原点 = -((192/2) + (-64)) * 0.2 = -6.4 米
- Y原点 = -(192/2) * 0.2 = -19.2 米
- Z原点 = -2.0 米（基于经验的合理值）
"""

import torch
import numpy as np
from typing import List, Tuple


def create_precise_point_labels_from_config(
    points_batch: torch.Tensor,
    anomaly_labels_batch: List,
    cfg,  # 配置对象
    device: torch.device = None
) -> torch.Tensor:
    """
    根据真实的体素异常标注和精确的坐标系变换，为每个点生成0/1标签。
    这是方案A的核心实现。

    Args:
        points_batch: 整个批次的点云张量 [B, N, 4] (x, y, z, intensity)
        anomaly_labels_batch: 整个批次的异常标签列表，每个元素是一个list of dicts
        cfg: 配置文件对象 (CfgNode)
        device: PyTorch设备

    Returns:
        一个形状为 [B, N] 的精确标签张量
    """
    if device is None:
        device = points_batch.device
    
    B, N, _ = points_batch.shape
    all_point_labels = torch.zeros(B, N, device=device)

    # 从config中获取体素化参数
    resolution = cfg.VOXEL.RESOLUTION  # 0.2
    grid_size = cfg.VOXEL.SIZE  # [192, 192, 64]
    offset_forward = cfg.BEV.OFFSET_FORWARD  # -64
    
    # 根据推导计算网格原点
    grid_origin_x = -((grid_size[0] / 2) + offset_forward) * resolution
    grid_origin_y = -(grid_size[1] / 2) * resolution
    grid_origin_z = -2.0  # 基于经验的合理值
    
    grid_origin = torch.tensor(
        [grid_origin_x, grid_origin_y, grid_origin_z], 
        device=device
    ).view(1, 1, 3)
    
    print(f"\n🗺️ 体素网格配置:")
    print(f"   网格大小: {grid_size}")
    print(f"   分辨率: {resolution}m")
    print(f"   前向偏移: {offset_forward}")
    print(f"   推导原点: [{grid_origin_x:.1f}, {grid_origin_y:.1f}, {grid_origin_z:.1f}]")
    print(f"   覆盖范围: X[{grid_origin_x:.1f}, {grid_origin_x + grid_size[0]*resolution:.1f}], "
          f"Y[{grid_origin_y:.1f}, {grid_origin_y + grid_size[1]*resolution:.1f}], "
          f"Z[{grid_origin_z:.1f}, {grid_origin_z + grid_size[2]*resolution:.1f}]")

    total_anomaly_points = 0
    total_normal_points = 0
    
    for i in range(B):  # 遍历batch中的每个样本
        single_points = points_batch[i]  # [N, 4]
        single_anomaly_label_list = anomaly_labels_batch[i]

        # 检查是否有异常
        if not single_anomaly_label_list:
            total_normal_points += N
            continue
        
        # 如果是字典而不是列表，转换为列表
        if isinstance(single_anomaly_label_list, dict):
            anomaly_is_alive = single_anomaly_label_list.get('anomaly_is_alive', 'false')
            if anomaly_is_alive == 'false':
                total_normal_points += N
                continue
            single_anomaly_label_list = single_anomaly_label_list.get('anomaly_coords', [])
        
        if not single_anomaly_label_list:
            total_normal_points += N
            continue

        # 1. 创建异常体素坐标的哈希集合以便快速查询
        anomaly_voxel_set = set()
        for label in single_anomaly_label_list:
            if isinstance(label, dict) and 'x' in label:
                vx = int(label['x'])
                vy = int(label['y'])
                vz = int(label['z'])
                anomaly_voxel_set.add((vx, vy, vz))
        
        if not anomaly_voxel_set:
            total_normal_points += N
            continue
        
        print(f"   样本 {i}: 异常体素数量 = {len(anomaly_voxel_set)}")

        # 2. 将所有点的真实世界坐标转换为体素索引
        point_coords_real = single_points[:, :3].unsqueeze(0)  # [1, N, 3]
        
        # 应用变换: (点坐标 - 网格原点) / 体素大小
        point_voxel_indices = torch.floor(
            (point_coords_real - grid_origin) / resolution
        ).long()  # [1, N, 3]
        
        # 3. 检查边界并为每个点生成标签
        point_voxel_indices_np = point_voxel_indices.squeeze(0).cpu().numpy()  # [N, 3]
        
        labels_list = []
        anomaly_count = 0
        
        for j in range(N):
            vx, vy, vz = point_voxel_indices_np[j]
            
            # 边界检查
            if 0 <= vx < grid_size[0] and 0 <= vy < grid_size[1] and 0 <= vz < grid_size[2]:
                if (vx, vy, vz) in anomaly_voxel_set:
                    labels_list.append(1.0)
                    anomaly_count += 1
                else:
                    labels_list.append(0.0)
            else:
                # 在网格外的点肯定是正常的
                labels_list.append(0.0)
        
        all_point_labels[i] = torch.tensor(labels_list, device=device)
        total_anomaly_points += anomaly_count
        total_normal_points += (N - anomaly_count)
        
        if anomaly_count > 0:
            print(f"   ✅ 样本 {i}: 检测到 {anomaly_count}/{N} ({100*anomaly_count/N:.2f}%) 个异常点")

    print(f"\n📊 批次标签统计:")
    print(f"   异常点: {total_anomaly_points}/{total_anomaly_points + total_normal_points} "
          f"({100*total_anomaly_points/(total_anomaly_points + total_normal_points):.2f}%)")
    print(f"   正常点: {total_normal_points}/{total_anomaly_points + total_normal_points} "
          f"({100*total_normal_points/(total_anomaly_points + total_normal_points):.2f}%)")

    return all_point_labels


def verify_coordinate_mapping(
    sample_points: np.ndarray,
    sample_voxel_coords: List[dict],
    cfg
) -> None:
    """
    验证坐标映射的正确性
    
    Args:
        sample_points: 点云坐标 [N, 3]
        sample_voxel_coords: 体素坐标列表
        cfg: 配置对象
    """
    resolution = cfg.VOXEL.RESOLUTION
    grid_size = cfg.VOXEL.SIZE
    offset_forward = cfg.BEV.OFFSET_FORWARD
    
    grid_origin_x = -((grid_size[0] / 2) + offset_forward) * resolution
    grid_origin_y = -(grid_size[1] / 2) * resolution
    grid_origin_z = -2.0
    
    print("\n🔍 坐标映射验证:")
    print(f"网格原点: ({grid_origin_x}, {grid_origin_y}, {grid_origin_z})")
    print(f"点云范围:")
    print(f"  X: [{sample_points[:, 0].min():.2f}, {sample_points[:, 0].max():.2f}]")
    print(f"  Y: [{sample_points[:, 1].min():.2f}, {sample_points[:, 1].max():.2f}]")
    print(f"  Z: [{sample_points[:, 2].min():.2f}, {sample_points[:, 2].max():.2f}]")
    
    # 验证几个点
    print("\n前5个点的映射:")
    for i in range(min(5, len(sample_points))):
        px, py, pz = sample_points[i]
        vx = int((px - grid_origin_x) / resolution)
        vy = int((py - grid_origin_y) / resolution)
        vz = int((pz - grid_origin_z) / resolution)
        print(f"  点 {i}: ({px:.2f}, {py:.2f}, {pz:.2f}) -> 体素 ({vx}, {vy}, {vz})")
    
    if sample_voxel_coords:
        print(f"\n异常体素坐标范围:")
        vx_coords = [v['x'] for v in sample_voxel_coords if 'x' in v]
        vy_coords = [v['y'] for v in sample_voxel_coords if 'y' in v]
        vz_coords = [v['z'] for v in sample_voxel_coords if 'z' in v]
        
        if vx_coords:
            print(f"  VX: [{min(vx_coords)}, {max(vx_coords)}]")
            print(f"  VY: [{min(vy_coords)}, {max(vy_coords)}]")
            print(f"  VZ: [{min(vz_coords)}, {max(vz_coords)}]")
            
            # 反向映射：体素中心对应的真实坐标
            print(f"\n前3个异常体素的真实世界坐标:")
            for i in range(min(3, len(sample_voxel_coords))):
                vx = sample_voxel_coords[i]['x']
                vy = sample_voxel_coords[i]['y']
                vz = sample_voxel_coords[i]['z']
                
                # 体素中心坐标
                cx = grid_origin_x + (vx + 0.5) * resolution
                cy = grid_origin_y + (vy + 0.5) * resolution
                cz = grid_origin_z + (vz + 0.5) * resolution
                
                print(f"  体素 ({vx}, {vy}, {vz}) -> 中心点 ({cx:.2f}, {cy:.2f}, {cz:.2f})")

