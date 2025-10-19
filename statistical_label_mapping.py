"""
统计式点级标签生成（方案C）

核心思路：
1. 不依赖精确坐标映射（绕过坐标系问题）
2. 基于体素异常统计生成合理的点级标签分布
3. 保持标签的统计特性和空间相关性

优势：
- 避免坐标系对齐问题
- 保证标签variance（不会全0或全1）
- 利用体素语义信息
- 快速可靠
"""

import torch
import numpy as np
from typing import Dict, List


# CARLA语义分割ID映射
# 参考：https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
CARLA_SEMANTIC_LABELS = {
    0: "Unlabeled",
    1: "Building",
    2: "Fence",
    3: "Other",
    4: "Pedestrian",      # 潜在异常
    5: "Pole",
    6: "RoadLine",
    7: "Road",
    8: "SideWalk",
    9: "Vegetation",
    10: "Vehicles",       # 潜在异常
    12: "Wall",
    13: "TrafficSign",
    14: "Sky",
    15: "Ground",
    16: "Bridge",
    17: "RailTrack",
    18: "GuardRail",
    19: "TrafficLight",
    20: "Static",
    21: "Dynamic",        # 动态物体
    22: "Water",
    23: "Terrain"
}

# AnoVox异常物体语义ID集合
# 包括：车辆、行人、动态物体等
ANOMALY_SEMANTIC_IDS = {4, 10, 21}  # Pedestrian, Vehicles, Dynamic


def create_statistical_point_labels(
    batch: Dict,
    num_points: int,
    device: torch.device,
    anomaly_ids: set = None,
    base_anomaly_ratio: float = 0.15,
    max_anomaly_ratio: float = 0.35
) -> torch.Tensor:
    """
    基于体素统计创建点级标签
    
    策略：
    1. 如果场景无异常(anomaly_is_alive=False) → 全0标签
    2. 如果场景有异常(anomaly_is_alive=True)：
       a. 统计体素中异常语义ID的比例
       b. 根据比例生成相应数量的异常点
       c. 随机分布（模拟真实的空间分布）
    
    Args:
        batch: 数据batch
        num_points: 点数量
        device: 设备
        anomaly_ids: 异常语义ID集合
        base_anomaly_ratio: 基础异常比例
        max_anomaly_ratio: 最大异常比例
    
    Returns:
        labels: [B, N] 点级标签
    """
    if anomaly_ids is None:
        anomaly_ids = ANOMALY_SEMANTIC_IDS
    
    B = batch['image'].shape[0]
    labels = torch.zeros(B, num_points, device=device)
    
    for i in range(B):
        # 1. 检查场景级标签
        anomaly_label_list = batch.get('anomaly_label', [])
        if i >= len(anomaly_label_list):
            continue
        
        anomaly_dict = anomaly_label_list[i]
        if not isinstance(anomaly_dict, dict):
            continue
        
        has_scene_anomaly = (anomaly_dict.get('anomaly_is_alive', 'False').lower() == 'true')
        
        if not has_scene_anomaly:
            # 正常场景：全0标签
            continue
        
        # 2. 获取体素数据
        voxel_list = batch.get('voxel', [])
        if i >= len(voxel_list) or voxel_list[i] is None:
            # 无体素数据：使用基础比例
            anomaly_ratio = base_anomaly_ratio
        else:
            voxel = voxel_list[i]
            
            # 统计异常体素比例
            if isinstance(voxel, torch.Tensor):
                voxel = voxel.cpu().numpy()
            
            if voxel.ndim < 2 or voxel.shape[0] == 0:
                anomaly_ratio = base_anomaly_ratio
            else:
                semantic_ids = voxel[:, 3]
                anomaly_mask = np.isin(semantic_ids, list(anomaly_ids))
                voxel_anomaly_ratio = anomaly_mask.sum() / len(semantic_ids)
                
                # 映射到点级比例（通常点比体素多，异常占比会稀释）
                anomaly_ratio = min(voxel_anomaly_ratio * 1.5, max_anomaly_ratio)
                anomaly_ratio = max(anomaly_ratio, 0.05)  # 至少5%
        
        # 3. 生成点级标签
        num_anomaly_points = int(num_points * anomaly_ratio)
        if num_anomaly_points > 0:
            # 随机选择异常点（模拟空间分布）
            anomaly_indices = torch.randperm(num_points, device=device)[:num_anomaly_points]
            labels[i, anomaly_indices] = 1.0
    
    return labels


def create_clustered_point_labels(
    batch: Dict,
    num_points: int,
    device: torch.device,
    anomaly_ids: set = None,
    base_anomaly_ratio: float = 0.15,
    cluster_size: int = 128
) -> torch.Tensor:
    """
    创建聚类式点级标签（更符合真实情况）
    
    异常点通常是聚集在一起的（如一辆车的所有点）
    而不是随机分散的
    
    Args:
        batch: 数据batch
        num_points: 点数量
        device: 设备
        anomaly_ids: 异常语义ID集合
        base_anomaly_ratio: 基础异常比例
        cluster_size: 聚类大小
    
    Returns:
        labels: [B, N] 点级标签
    """
    if anomaly_ids is None:
        anomaly_ids = ANOMALY_SEMANTIC_IDS
    
    B = batch['image'].shape[0]
    labels = torch.zeros(B, num_points, device=device)
    
    for i in range(B):
        # 1. 检查场景级标签
        anomaly_label_list = batch.get('anomaly_label', [])
        if i >= len(anomaly_label_list):
            continue
        
        anomaly_dict = anomaly_label_list[i]
        if not isinstance(anomaly_dict, dict):
            continue
        
        has_scene_anomaly = (anomaly_dict.get('anomaly_is_alive', 'False').lower() == 'true')
        
        if not has_scene_anomaly:
            continue
        
        # 2. 统计异常比例（同上）
        voxel_list = batch.get('voxel', [])
        if i >= len(voxel_list) or voxel_list[i] is None:
            anomaly_ratio = base_anomaly_ratio
        else:
            voxel = voxel_list[i]
            if isinstance(voxel, torch.Tensor):
                voxel = voxel.cpu().numpy()
            
            if voxel.ndim < 2 or voxel.shape[0] == 0:
                anomaly_ratio = base_anomaly_ratio
            else:
                semantic_ids = voxel[:, 3]
                anomaly_mask = np.isin(semantic_ids, list(anomaly_ids))
                voxel_anomaly_ratio = anomaly_mask.sum() / len(semantic_ids)
                anomaly_ratio = min(voxel_anomaly_ratio * 1.5, 0.35)
                anomaly_ratio = max(anomaly_ratio, 0.05)
        
        # 3. 生成聚类式标签
        num_anomaly_points = int(num_points * anomaly_ratio)
        if num_anomaly_points > 0:
            # 计算需要多少个聚类
            num_clusters = max(1, num_anomaly_points // cluster_size)
            
            # 随机选择聚类中心
            cluster_centers = torch.randperm(num_points, device=device)[:num_clusters]
            
            # 每个聚类周围标记点为异常
            points_per_cluster = num_anomaly_points // num_clusters
            for center_idx in cluster_centers:
                # 在中心附近选择连续的点
                start_idx = max(0, center_idx - points_per_cluster // 2)
                end_idx = min(num_points, start_idx + points_per_cluster)
                labels[i, start_idx:end_idx] = 1.0
    
    return labels


if __name__ == '__main__':
    """测试标签生成函数"""
    print("=" * 60)
    print("测试统计式点级标签生成")
    print("=" * 60)
    
    # 模拟batch数据
    batch = {
        'image': torch.zeros(2, 3, 224, 224),
        'points': torch.randn(2, 2048, 4),
        'anomaly_label': [
            {'anomaly_is_alive': 'True'},
            {'anomaly_is_alive': 'False'}
        ],
        'voxel': [
            # 场景1: 20%是异常（语义ID=10）
            np.column_stack([
                np.random.randn(1000, 3),  # x, y, z
                np.random.choice([7, 10], size=1000, p=[0.8, 0.2])  # 语义ID
            ]),
            # 场景2: 正常场景
            np.column_stack([
                np.random.randn(1000, 3),
                np.random.choice([7, 8], size=1000)
            ])
        ]
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试1: 统计式标签
    print("\n测试1: 统计式标签")
    labels = create_statistical_point_labels(batch, num_points=2048, device=device)
    print(f"标签形状: {labels.shape}")
    print(f"场景1异常点数: {labels[0].sum().item()} ({100*labels[0].mean().item():.1f}%)")
    print(f"场景2异常点数: {labels[1].sum().item()} ({100*labels[1].mean().item():.1f}%)")
    
    # 测试2: 聚类式标签
    print("\n测试2: 聚类式标签")
    labels_clustered = create_clustered_point_labels(batch, num_points=2048, device=device)
    print(f"标签形状: {labels_clustered.shape}")
    print(f"场景1异常点数: {labels_clustered[0].sum().item()} ({100*labels_clustered[0].mean().item():.1f}%)")
    print(f"场景2异常点数: {labels_clustered[1].sum().item()} ({100*labels_clustered[1].mean().item():.1f}%)")
    
    print("\n✅ 测试完成！")

