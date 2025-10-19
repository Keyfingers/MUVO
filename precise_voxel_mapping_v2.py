"""
🎯 精确的体素-点标签映射（方案A完整实现 V2）

根据config.py中的参数推导出的精确坐标系变换：
- VOXEL.SIZE = [192, 192, 64]
- VOXEL.RESOLUTION = 0.2
- BEV.OFFSET_FORWARD = -64

推导出的体素网格原点：
- X原点 = -((192/2) + (-64)) * 0.2 = -6.4 米
- Y原点 = -(192/2) * 0.2 = -19.2 米  
- Z原点 = -2.0 米（基于经验的合理值）

** 重要：AnoVox体素数据格式是 [N, 4] (vx, vy, vz, semantic_id) **
"""

import torch
import numpy as np
from typing import List, Set


# CARLA语义ID定义（来自CARLA文档）
CARLA_ANOMALY_SEMANTIC_IDS = {
    4,   # Pedestrian (person)
    10,  # Vehicle  
    12,  # Rider
    18,  # TrafficLight
    19,  # TrafficSign
    # 其他可能的异常物体
    5,   # Pole
    6,   # TrafficLight
    7,   # TrafficSign
}


def create_precise_point_labels_from_voxels(
    points_batch: torch.Tensor,
    voxel_data_list: List[np.ndarray],  # 每个样本的体素数据 [N, 4]
    anomaly_labels_batch: List[dict],
    cfg,
    anomaly_semantic_ids: Set[int] = None,
    device: torch.device = None
) -> torch.Tensor:
    """
    根据体素数据中的semantic_id和精确坐标映射生成点级标签
    
    Args:
        points_batch: [B, N, 4] 点云批次
        voxel_data_list: 长度为B的列表，每个元素是 [M, 4] 的numpy数组 (vx, vy, vz, semantic_id)
        anomaly_labels_batch: 场景级异常标签（用于判断该场景是否有异常）
        cfg: 配置对象
        anomaly_semantic_ids: 哪些semantic_id被认为是异常（默认使用CARLA标准）
        device: PyTorch设备
    
    Returns:
        [B, N] 标签张量
    """
    if device is None:
        device = points_batch.device
    
    if anomaly_semantic_ids is None:
        anomaly_semantic_ids = CARLA_ANOMALY_SEMANTIC_IDS
    
    B, N, _ = points_batch.shape
    all_point_labels = torch.zeros(B, N, device=device)
    
    # 从config中获取体素化参数
    resolution = cfg.VOXEL.RESOLUTION  # 0.2
    grid_size = cfg.VOXEL.SIZE  # [192, 192, 64]
    offset_forward = cfg.BEV.OFFSET_FORWARD  # -64
    
    # 计算网格原点
    grid_origin_x = -((grid_size[0] / 2) + offset_forward) * resolution
    grid_origin_y = -(grid_size[1] / 2) * resolution
    grid_origin_z = -2.0
    
    grid_origin = torch.tensor(
        [grid_origin_x, grid_origin_y, grid_origin_z],
        device=device
    ).view(1, 1, 3)
    
    total_anomaly_points = 0
    total_normal_points = 0
    samples_with_anomalies = 0
    
    for i in range(B):
        # 检查该场景是否有异常
        anomaly_label = anomaly_labels_batch[i] if i < len(anomaly_labels_batch) else {}
        anomaly_is_alive = anomaly_label.get('anomaly_is_alive', 'False')
        
        # 转换为布尔值
        if isinstance(anomaly_is_alive, str):
            has_anomaly = (anomaly_is_alive.lower() == 'true')
        else:
            has_anomaly = bool(anomaly_is_alive)
        
        # 如果没有异常，跳过
        if not has_anomaly:
            total_normal_points += N
            continue
        
        # 获取该样本的体素数据
        if i >= len(voxel_data_list) or voxel_data_list[i] is None or len(voxel_data_list[i]) == 0:
            total_normal_points += N
            continue
        
        voxel_data = voxel_data_list[i]  # [M, 4]
        
        # 提取异常体素坐标
        # voxel_data: (vx, vy, vz, semantic_id)
        semantic_ids = voxel_data[:, 3]
        anomaly_mask = np.isin(semantic_ids, list(anomaly_semantic_ids))
        
        if not anomaly_mask.any():
            # 没有异常体素
            total_normal_points += N
            continue
        
        # 构建异常体素坐标集合
        anomaly_voxels = voxel_data[anomaly_mask, :3]  # [K, 3] (vx, vy, vz)
        anomaly_voxel_set = set(
            (int(vx), int(vy), int(vz))
            for vx, vy, vz in anomaly_voxels
        )
        
        # 将点云坐标转换为体素索引
        single_points = points_batch[i]  # [N, 4]
        point_coords_real = single_points[:, :3].unsqueeze(0)  # [1, N, 3]
        
        # 应用变换
        point_voxel_indices = torch.floor(
            (point_coords_real - grid_origin) / resolution
        ).long()  # [1, N, 3]
        
        # 生成标签
        point_voxel_indices_np = point_voxel_indices.squeeze(0).cpu().numpy()
        
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
                labels_list.append(0.0)
        
        all_point_labels[i] = torch.tensor(labels_list, device=device)
        total_anomaly_points += anomaly_count
        total_normal_points += (N - anomaly_count)
        
        if anomaly_count > 0:
            samples_with_anomalies += 1

    return all_point_labels

