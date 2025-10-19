"""
🎯 精确的体素-点标签映射 V3 (最终版)

基于导师的关键发现：
- EV_POSITION = [32, 96, 12] 是 Ego Vehicle（自车）在体素网格中的位置
- 这是一个明确的锚点：真实世界(0,0,0) 对应体素索引(32,96,12)

推导过程：
- real_coord = grid_origin + voxel_index * resolution
- 0 = grid_origin_x + 32 * 0.2  =>  grid_origin_x = -6.4
- 0 = grid_origin_y + 96 * 0.2  =>  grid_origin_y = -19.2
- 0 = grid_origin_z + 12 * 0.2  =>  grid_origin_z = -2.4

最终网格原点：[-6.4, -19.2, -2.4] 米
"""

import torch
import numpy as np
from typing import List, Set, Dict


# CARLA语义ID定义（基于CARLA文档）
# 这些是可能的异常物体
CARLA_ANOMALY_SEMANTIC_IDS = {
    4,   # Pedestrian
    10,  # Vehicle (最重要的异常)
    12,  # Rider
    18,  # TrafficLight
    19,  # TrafficSign
    5,   # Pole
    6,   # TrafficLight (duplicate?)
    7,   # TrafficSign (duplicate?)
    14,  # 从数据中观察到的ID
    15,  # 从数据中观察到的ID
}


def create_precise_point_labels_v3(
    points_batch: torch.Tensor,
    voxel_data_list: List[np.ndarray],
    anomaly_labels_batch: List[Dict],
    cfg,
    anomaly_semantic_ids: Set[int] = None,
    device: torch.device = None
) -> torch.Tensor:
    """
    精确的点-体素标签映射 V3 - 基于EV_POSITION锚点。
    这是方案A的最终实现。

    Args:
        points_batch: [B, N, 4] 点云批次
        voxel_data_list: 长度为B的列表，每个元素是 [M, 4] 的numpy数组 (vx, vy, vz, semantic_id)
        anomaly_labels_batch: 场景级异常标签（用于判断该场景是否有异常）
        cfg: 配置文件对象
        anomaly_semantic_ids: 哪些semantic_id被认为是异常（默认使用CARLA标准）
        device: PyTorch设备

    Returns:
        [B, N] 精确的标签张量
    """
    if device is None:
        device = points_batch.device
    
    if anomaly_semantic_ids is None:
        anomaly_semantic_ids = CARLA_ANOMALY_SEMANTIC_IDS
    
    B, N, _ = points_batch.shape
    all_point_labels = torch.zeros(B, N, device=device)

    # 从config中获取关键参数
    resolution = cfg.VOXEL.RESOLUTION  # 0.2
    ev_position = cfg.VOXEL.EV_POSITION  # [32, 96, 12]

    # 🎯 V3核心：基于EV_POSITION计算网格原点
    # 锚点：真实世界(0,0,0) 对应体素索引(32,96,12)
    grid_origin_x = 0 - ev_position[0] * resolution  # -6.4
    grid_origin_y = 0 - ev_position[1] * resolution  # -19.2
    grid_origin_z = 0 - ev_position[2] * resolution  # -2.4
    
    grid_origin = torch.tensor(
        [grid_origin_x, grid_origin_y, grid_origin_z],
        device=device
    ).view(1, 1, 3)

    print(f"\n🗺️ V3体素网格配置 (基于EV_POSITION锚点):")
    print(f"   EV位置: {ev_position}")
    print(f"   分辨率: {resolution}m")
    print(f"   网格原点: [{grid_origin_x:.1f}, {grid_origin_y:.1f}, {grid_origin_z:.1f}]")

    total_anomaly_points = 0
    total_normal_points = 0
    samples_with_anomalies = 0

    for i in range(B):  # 遍历batch中的每个样本
        # 检查该场景是否有异常
        anomaly_label = anomaly_labels_batch[i] if i < len(anomaly_labels_batch) else {}
        anomaly_is_alive = anomaly_label.get('anomaly_is_alive', 'False')
        
        # 转换为布尔值
        if isinstance(anomaly_is_alive, str):
            has_anomaly = (anomaly_is_alive.lower() == 'true')
        else:
            has_anomaly = bool(anomaly_is_alive)
        
        # 如果场景没有异常，跳过
        if not has_anomaly:
            total_normal_points += N
            continue
        
        # 获取该样本的体素数据
        if i >= len(voxel_data_list) or voxel_data_list[i] is None or len(voxel_data_list[i]) == 0:
            total_normal_points += N
            continue
        
        voxel_data = voxel_data_list[i]  # [M, 4]

        # 1. 创建异常体素坐标的哈希集合
        semantic_ids = voxel_data[:, 3]
        anomaly_mask = np.isin(semantic_ids, list(anomaly_semantic_ids))
        
        if not anomaly_mask.any():
            # 场景有异常但体素中没有对应的语义ID
            total_normal_points += N
            continue
        
        # 构建异常体素坐标集合
        anomaly_voxels = voxel_data[anomaly_mask, :3]  # [K, 3] (vx, vy, vz)
        anomaly_voxel_set = set(
            (int(vx), int(vy), int(vz))
            for vx, vy, vz in anomaly_voxels
        )
        
        print(f"   样本 {i}: 异常体素数量 = {len(anomaly_voxel_set)}")

        # 2. 将点云坐标转换为体素索引
        single_points = points_batch[i]  # [N, 4]
        point_coords_real = single_points[:, :3].unsqueeze(0)  # [1, N, 3]
        
        # 应用变换：voxel_index = floor((real_coord - grid_origin) / resolution)
        point_voxel_indices = torch.floor(
            (point_coords_real - grid_origin) / resolution
        ).long().squeeze(0)  # [N, 3]

        # 3. 生成精确标签
        point_voxel_indices_np = point_voxel_indices.cpu().numpy()
        
        labels_list = []
        anomaly_count = 0
        
        for j in range(N):
            vx, vy, vz = point_voxel_indices_np[j]
            
            if (vx, vy, vz) in anomaly_voxel_set:
                labels_list.append(1.0)
                anomaly_count += 1
            else:
                labels_list.append(0.0)
        
        all_point_labels[i] = torch.tensor(labels_list, device=device)
        total_anomaly_points += anomaly_count
        total_normal_points += (N - anomaly_count)
        
        if anomaly_count > 0:
            samples_with_anomalies += 1
            print(f"   ✅ 样本 {i}: {anomaly_count}/{N} ({100*anomaly_count/N:.2f}%) 异常点")

    print(f"\n📊 批次标签统计:")
    print(f"   有异常的样本: {samples_with_anomalies}/{B}")
    if total_anomaly_points + total_normal_points > 0:
        print(f"   异常点: {total_anomaly_points:,}/{total_anomaly_points + total_normal_points:,} "
              f"({100*total_anomaly_points/(total_anomaly_points + total_normal_points):.2f}%)")
        print(f"   正常点: {total_normal_points:,}/{total_anomaly_points + total_normal_points:,} "
              f"({100*total_normal_points/(total_anomaly_points + total_normal_points):.2f}%)")

    return all_point_labels


def visualize_point_labels(
    points: np.ndarray,
    labels: np.ndarray,
    save_path: str = None
) -> None:
    """
    可视化点云和标签（用于验证）
    
    Args:
        points: [N, 3] 点云坐标
        labels: [N] 标签 (0/1)
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 俯视图 (XY平面)
    ax1 = fig.add_subplot(131)
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    ax1.scatter(points[normal_mask, 0], points[normal_mask, 1], 
                c='blue', s=1, alpha=0.3, label='Normal')
    ax1.scatter(points[anomaly_mask, 0], points[anomaly_mask, 1], 
                c='red', s=5, alpha=0.8, label='Anomaly')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Top View (XY)')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # 2. 侧视图 (XZ平面)
    ax2 = fig.add_subplot(132)
    ax2.scatter(points[normal_mask, 0], points[normal_mask, 2], 
                c='blue', s=1, alpha=0.3, label='Normal')
    ax2.scatter(points[anomaly_mask, 0], points[anomaly_mask, 2], 
                c='red', s=5, alpha=0.8, label='Anomaly')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Side View (XZ)')
    ax2.legend()
    ax2.grid(True)
    
    # 3. 3D视图
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(points[normal_mask, 0], points[normal_mask, 1], points[normal_mask, 2],
                c='blue', s=0.5, alpha=0.2, label='Normal')
    ax3.scatter(points[anomaly_mask, 0], points[anomaly_mask, 1], points[anomaly_mask, 2],
                c='red', s=3, alpha=0.8, label='Anomaly')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_title('3D View')
    ax3.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 可视化已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


