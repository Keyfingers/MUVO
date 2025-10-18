"""
AnoVox数据集适配器
用于将AnoVox格式数据加载到MUVO训练框架中

AnoVox数据格式:
- Scenario_xxx/
  - RGB-CAM(x,y,z)(...)_id/
    - *.png (图像帧)
  - LIDAR(x,y,z)(...)_id/
    - *.ply (点云帧)
  - VOXEL_GRID/
    - *.npz (体素网格)
  - ANOMALY/
    - *.json (异常标注)
  - sensor_setup.json
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import open3d as o3d


class AnoVoxDataset(Dataset):
    """
    AnoVox数据集加载器
    将AnoVox格式转换为MUVO可用的格式
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        sequence_length: int = 1,
        transform=None,
        load_voxel: bool = True,
        load_anomaly_labels: bool = True
    ):
        """
        Args:
            data_root: AnoVox数据集根目录
            split: 'train' 或 'val' 或 'test'
            sequence_length: 时间序列长度（目前支持单帧=1）
            transform: 数据增强
            load_voxel: 是否加载体素数据
            load_anomaly_labels: 是否加载异常标注
        """
        self.data_root = Path(data_root)
        self.split = split
        self.sequence_length = sequence_length
        self.transform = transform
        self.load_voxel = load_voxel
        self.load_anomaly_labels = load_anomaly_labels
        
        # 扫描所有场景
        self.scenarios = self._scan_scenarios()
        
        # 构建样本索引
        self.samples = self._build_sample_index()
        
        print(f"[AnoVoxDataset] 加载 {split} 集: {len(self.scenarios)} 场景, {len(self.samples)} 样本")
    
    def _scan_scenarios(self) -> List[Path]:
        """扫描数据集中的所有场景文件夹"""
        scenarios = []
        
        # AnoVox数据集通常按Town组织
        for scenario_dir in sorted(self.data_root.glob("Scenario_*")):
            if scenario_dir.is_dir():
                scenarios.append(scenario_dir)
        
        return scenarios
    
    def _build_sample_index(self) -> List[Dict]:
        """
        构建样本索引
        每个样本包含：场景路径、帧ID、相关文件路径
        """
        samples = []
        
        for scenario_path in self.scenarios:
            # 读取传感器配置
            sensor_config_path = scenario_path / "sensor_setup.json"
            if not sensor_config_path.exists():
                print(f"[Warning] 跳过场景 {scenario_path.name}: 缺少sensor_setup.json")
                continue
            
            with open(sensor_config_path, 'r') as f:
                sensor_config = json.load(f)
            
            # 查找RGB相机和LiDAR目录
            rgb_dir = None
            lidar_dir = None
            
            for item in scenario_path.iterdir():
                if item.is_dir():
                    name = item.name
                    if name.startswith("RGB-CAM"):
                        rgb_dir = item
                    elif name.startswith("LIDAR") and not name.startswith("SEMANTIC-LIDAR"):
                        lidar_dir = item
            
            if rgb_dir is None or lidar_dir is None:
                print(f"[Warning] 跳过场景 {scenario_path.name}: 缺少RGB或LiDAR数据")
                continue
            
            # 获取所有帧
            rgb_files = sorted(rgb_dir.glob("*.png"))
            
            for rgb_file in rgb_files:
                # 提取帧ID
                frame_id = rgb_file.stem.split('_')[-1]
                
                # 构建对应的点云文件路径（AnoVox使用.npy格式）
                lidar_files = list(lidar_dir.glob(f"*_{frame_id}.npy"))
                if len(lidar_files) == 0:
                    # 尝试.ply格式作为备选
                    lidar_files = list(lidar_dir.glob(f"*_{frame_id}.ply"))
                    if len(lidar_files) == 0:
                        continue
                lidar_file = lidar_files[0]
                
                # 构建体素和异常标注路径（如果需要）
                voxel_file = None
                anomaly_file = None
                
                if self.load_voxel:
                    voxel_dir = scenario_path / "VOXEL_GRID"
                    if voxel_dir.exists():
                        voxel_files = list(voxel_dir.glob(f"*_{frame_id}.npz"))
                        if len(voxel_files) > 0:
                            voxel_file = voxel_files[0]
                
                if self.load_anomaly_labels:
                    anomaly_dir = scenario_path / "ANOMALY"
                    if anomaly_dir.exists():
                        anomaly_files = list(anomaly_dir.glob(f"*_{frame_id}.json"))
                        if len(anomaly_files) > 0:
                            anomaly_file = anomaly_files[0]
                
                sample = {
                    'scenario': scenario_path.name,
                    'frame_id': frame_id,
                    'rgb_path': rgb_file,
                    'lidar_path': lidar_file,
                    'voxel_path': voxel_file,
                    'anomaly_path': anomaly_file,
                    'sensor_config': sensor_config
                }
                
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        加载单个样本
        
        Returns:
            Dict包含:
                - image: [3, H, W] RGB图像
                - points: [N, 4] 点云 (x, y, z, intensity/reflectance)
                - voxel: [X, Y, Z] 体素网格（可选）
                - anomaly_label: 异常标注（可选）
                - metadata: 元数据
        """
        sample_info = self.samples[idx]
        
        # 1. 加载RGB图像
        image = Image.open(sample_info['rgb_path']).convert('RGB')
        image = np.array(image)  # [H, W, 3]
        
        # 2. 加载点云数据
        lidar_path = sample_info['lidar_path']
        
        if str(lidar_path).endswith('.npy'):
            # AnoVox格式: numpy数组 [N, 3或4]
            points = np.load(lidar_path)  # [N, 3] or [N, 4]
            
            # 确保有4个通道 (x, y, z, intensity)
            if points.shape[1] == 3:
                # 如果只有3维，添加强度（用距离作为替代）
                intensity = np.linalg.norm(points, axis=1, keepdims=True)  # [N, 1]
                points = np.concatenate([points, intensity], axis=1)  # [N, 4]
        else:
            # PLY格式
            pcd = o3d.io.read_point_cloud(str(lidar_path))
            points = np.asarray(pcd.points)  # [N, 3]
            
            # 尝试获取强度信息
            if pcd.colors is not None and len(pcd.colors) > 0:
                intensity = np.asarray(pcd.colors)[:, 0:1]  # [N, 1]
            else:
                intensity = np.linalg.norm(points, axis=1, keepdims=True)  # [N, 1]
            
            points = np.concatenate([points, intensity], axis=1)  # [N, 4]
        
        # 3. 加载体素数据（如果存在）
        voxel = None
        if self.load_voxel and sample_info['voxel_path'] is not None:
            try:
                voxel_data = np.load(sample_info['voxel_path'])
                # AnoVox体素格式可能是 'voxel_grid' 或 'occupancy'
                if 'voxel_grid' in voxel_data:
                    voxel = voxel_data['voxel_grid']
                elif 'occupancy' in voxel_data:
                    voxel = voxel_data['occupancy']
                else:
                    voxel = voxel_data[voxel_data.files[0]]  # 取第一个数组
            except Exception as e:
                print(f"[Warning] 加载体素失败: {e}")
        
        # 4. 加载异常标注（如果存在）
        anomaly_label = None
        if self.load_anomaly_labels and sample_info['anomaly_path'] is not None:
            try:
                with open(sample_info['anomaly_path'], 'r') as f:
                    anomaly_label = json.load(f)
            except Exception as e:
                print(f"[Warning] 加载异常标注失败: {e}")
        
        # 5. 构建返回字典（适配MUVO格式）
        batch = {
            # 图像数据
            'image': torch.from_numpy(image).permute(2, 0, 1).float(),  # [3, H, W]
            
            # 点云数据
            'points': torch.from_numpy(points).float(),  # [N, 4]
            
            # 元数据
            'scenario': sample_info['scenario'],
            'frame_id': sample_info['frame_id'],
            'town': sample_info['scenario'].split('_')[0] if '_' in sample_info['scenario'] else 'Unknown',
        }
        
        # 添加体素数据
        if voxel is not None:
            batch['voxel'] = torch.from_numpy(voxel).float()
        
        # 添加异常标注
        if anomaly_label is not None:
            batch['anomaly_label'] = anomaly_label
        
        # 应用数据增强
        if self.transform is not None:
            batch = self.transform(batch)
        
        return batch
    
    def get_sample_info(self, idx: int) -> Dict:
        """获取样本的元信息（不加载实际数据）"""
        return self.samples[idx]


def collate_fn(batch):
    """
    自定义collate函数 - 处理不同大小的点云
    """
    import torch
    
    # 分离不同类型的数据
    images = torch.stack([item['image'] for item in batch])
    
    # 点云需要填充到相同大小
    points_list = [item['points'] for item in batch]
    max_points = max([p.shape[0] for p in points_list])
    
    # 填充点云
    padded_points = []
    for points in points_list:
        if points.shape[0] < max_points:
            # 填充零向量
            padding = torch.zeros((max_points - points.shape[0], points.shape[1]))
            points = torch.cat([points, padding], dim=0)
        padded_points.append(points)
    
    points = torch.stack(padded_points)
    
    # 其他元数据
    result = {
        'image': images,
        'points': points,
        'scenario': [item['scenario'] for item in batch],
        'frame_id': [item['frame_id'] for item in batch],
        'town': [item['town'] for item in batch],
    }
    
    # 可选字段
    if 'voxel' in batch[0]:
        result['voxel'] = torch.stack([item['voxel'] for item in batch])
    
    if 'anomaly_label' in batch[0]:
        result['anomaly_label'] = [item['anomaly_label'] for item in batch]
    
    return result


def create_anovox_dataloader(
    data_root: str,
    split: str = 'train',
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    **kwargs
):
    """
    创建AnoVox数据加载器的便捷函数
    
    Args:
        data_root: AnoVox数据集根目录
        split: 'train', 'val', 'test'
        batch_size: 批次大小
        num_workers: 数据加载线程数
        shuffle: 是否打乱
        **kwargs: 传递给AnoVoxDataset的其他参数
    
    Returns:
        DataLoader实例
    """
    from torch.utils.data import DataLoader
    
    dataset = AnoVoxDataset(
        data_root=data_root,
        split=split,
        **kwargs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        collate_fn=collate_fn  # 使用自定义collate函数
    )
    
    return dataloader


if __name__ == '__main__':
    # 测试代码
    print("=" * 60)
    print("测试AnoVox数据加载器")
    print("=" * 60)
    
    data_root = "/root/autodl-tmp/datasets/AnoVox/AnoVox_Dynamic_Mono_Town07"
    
    # 创建数据集
    dataset = AnoVoxDataset(
        data_root=data_root,
        split='train',
        load_voxel=True,
        load_anomaly_labels=True
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    if len(dataset) > 0:
        # 测试加载第一个样本
        print("\n测试加载第一个样本...")
        sample = dataset[0]
        
        print(f"\n样本内容:")
        print(f"  - 图像形状: {sample['image'].shape}")
        print(f"  - 点云形状: {sample['points'].shape}")
        print(f"  - 场景: {sample['scenario']}")
        print(f"  - 帧ID: {sample['frame_id']}")
        
        if 'voxel' in sample:
            print(f"  - 体素形状: {sample['voxel'].shape}")
        
        if 'anomaly_label' in sample:
            print(f"  - 异常标注: {sample['anomaly_label']}")
        
        print("\n✅ 数据加载测试成功！")

