"""
跨模态注意力融合模块 - 核心创新实现
Cross-Modal Attention Fusion Module - Core Innovation Implementation

这个模块实现了您方案中的核心创新：
1. 特征空间对齐：将图像特征根据映射关系赋予对应体素
2. 轻量级跨模态注意力：点云特征作为Query，图像特征作为Key/Value
3. 增强的点云特征输出：融合了图像纹理与上下文信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class CrossModalAttention(nn.Module):
    """
    轻量级跨模态注意力模块
    Lightweight Cross-Modal Attention Block
    
    输入：
    - F_pc: 点云特征 (作为Query)
    - F_img_aligned: 对齐后的图像特征 (作为Key/Value)
    
    输出：
    - F_enhanced_pc: 增强的点云特征，融合了图像信息
    """
    
    def __init__(self, 
                 pc_feature_dim: int = 64,
                 img_feature_dim: int = 64, 
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.pc_feature_dim = pc_feature_dim
        self.img_feature_dim = img_feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # 特征维度对齐
        self.pc_projection = nn.Linear(pc_feature_dim, hidden_dim)
        self.img_projection = nn.Linear(img_feature_dim, hidden_dim)
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, pc_feature_dim)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(pc_feature_dim)
        self.norm2 = nn.LayerNorm(pc_feature_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(pc_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pc_feature_dim),
            nn.Dropout(dropout)
        )
        
        # 残差连接的权重
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, 
                pc_features: torch.Tensor, 
                img_features: torch.Tensor,
                pc_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            pc_features: 点云特征 [B, N_pc, pc_feature_dim]
            img_features: 图像特征 [B, N_img, img_feature_dim] 
            pc_mask: 点云掩码 [B, N_pc] (可选)
            
        Returns:
            enhanced_pc_features: 增强的点云特征 [B, N_pc, pc_feature_dim]
        """
        B, N_pc, _ = pc_features.shape
        _, N_img, _ = img_features.shape
        
        # 特征投影到统一维度
        pc_proj = self.pc_projection(pc_features)  # [B, N_pc, hidden_dim]
        img_proj = self.img_projection(img_features)  # [B, N_img, hidden_dim]
        
        # 多头注意力：点云作为Query，图像作为Key/Value
        attn_output, attn_weights = self.attention(
            query=pc_proj,
            key=img_proj, 
            value=img_proj,
            key_padding_mask=None,  # 可以在这里添加图像掩码
            need_weights=False
        )
        
        # 残差连接和层归一化
        enhanced_features = self.norm1(pc_features + self.alpha * self.output_projection(attn_output))
        
        # 前馈网络
        ffn_output = self.ffn(enhanced_features)
        enhanced_features = self.norm2(enhanced_features + ffn_output)
        
        return enhanced_features


class FeatureAlignment(nn.Module):
    """
    特征空间对齐模块
    Feature Space Alignment Module
    
    将图像特征根据投影关系映射到体素空间
    """
    
    def __init__(self, 
                 img_feature_dim: int = 64,
                 voxel_size: Tuple[int, int, int] = (192, 192, 64),
                 alignment_method: str = 'nearest'):
        super().__init__()
        
        self.img_feature_dim = img_feature_dim
        self.voxel_size = voxel_size
        self.alignment_method = alignment_method
        
        # 特征维度调整
        self.feature_adapter = nn.Conv2d(img_feature_dim, img_feature_dim, 1)
        
    def forward(self, 
                img_features: torch.Tensor,
                projection_matrix: torch.Tensor,
                voxel_coords: torch.Tensor) -> torch.Tensor:
        """
        将图像特征对齐到体素空间
        
        Args:
            img_features: 图像特征 [B, C, H, W]
            projection_matrix: 投影矩阵 [B, 3, 4] 或 [B, 4, 4]
            voxel_coords: 体素坐标 [B, N_voxel, 3] (x, y, z)
            
        Returns:
            aligned_features: 对齐后的特征 [B, N_voxel, C]
        """
        B, C, H, W = img_features.shape
        N_voxel = voxel_coords.shape[1]
        
        # 特征适配
        img_features = self.feature_adapter(img_features)
        
        # 将体素坐标投影到图像平面
        # 这里需要根据具体的投影关系实现
        # 简化版本：使用最近邻插值
        if self.alignment_method == 'nearest':
            aligned_features = self._nearest_neighbor_alignment(
                img_features, voxel_coords, projection_matrix
            )
        else:
            raise NotImplementedError(f"Alignment method {self.alignment_method} not implemented")
            
        return aligned_features
    
    def _nearest_neighbor_alignment(self, 
                                   img_features: torch.Tensor,
                                   voxel_coords: torch.Tensor, 
                                   projection_matrix: torch.Tensor) -> torch.Tensor:
        """
        最近邻对齐方法
        """
        B, C, H, W = img_features.shape
        N_voxel = voxel_coords.shape[1]
        
        # 简化的投影：假设体素坐标直接对应图像坐标
        # 实际实现中需要根据具体的相机参数和体素化参数进行投影
        x_coords = torch.clamp(voxel_coords[:, :, 0], 0, W-1).long()
        y_coords = torch.clamp(voxel_coords[:, :, 1], 0, H-1).long()
        
        # 批量索引
        batch_indices = torch.arange(B, device=img_features.device).unsqueeze(1).expand(-1, N_voxel)
        
        # 提取对应位置的特征
        aligned_features = img_features[batch_indices, :, y_coords, x_coords]  # [B, N_voxel, C]
        
        return aligned_features


class VoxelFeatureExtractor(nn.Module):
    """
    体素特征提取器
    从体素化点云中提取特征
    """
    
    def __init__(self, 
                 input_dim: int = 4,  # x, y, z, intensity
                 feature_dim: int = 64,
                 voxel_size: Tuple[int, int, int] = (192, 192, 64)):
        super().__init__()
        
        self.voxel_size = voxel_size
        self.feature_dim = feature_dim
        
        # 3D卷积特征提取
        self.conv3d_layers = nn.Sequential(
            nn.Conv3d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
    def forward(self, voxel_data: torch.Tensor) -> torch.Tensor:
        """
        从体素数据中提取特征
        
        Args:
            voxel_data: 体素数据 [B, input_dim, X, Y, Z]
            
        Returns:
            features: 提取的特征 [B, feature_dim, X, Y, Z]
        """
        features = self.conv3d_layers(voxel_data)
        return features


class CrossModalFusionModule(nn.Module):
    """
    跨模态融合模块 - 整合所有组件
    Cross-Modal Fusion Module - Integrates all components
    """
    
    def __init__(self,
                 pc_feature_dim: int = 64,
                 img_feature_dim: int = 64,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 voxel_size: Tuple[int, int, int] = (192, 192, 64),
                 dropout: float = 0.1):
        super().__init__()
        
        # 特征对齐模块
        self.feature_alignment = FeatureAlignment(
            img_feature_dim=img_feature_dim,
            voxel_size=voxel_size
        )
        
        # 跨模态注意力模块
        self.cross_modal_attention = CrossModalAttention(
            pc_feature_dim=pc_feature_dim,
            img_feature_dim=img_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 特征融合权重
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self,
                pc_features: torch.Tensor,
                img_features: torch.Tensor,
                voxel_coords: torch.Tensor,
                projection_matrix: torch.Tensor) -> torch.Tensor:
        """
        跨模态融合前向传播
        
        Args:
            pc_features: 点云特征 [B, N_pc, pc_feature_dim]
            img_features: 图像特征 [B, C, H, W]
            voxel_coords: 体素坐标 [B, N_voxel, 3]
            projection_matrix: 投影矩阵 [B, 3, 4]
            
        Returns:
            enhanced_features: 增强的特征 [B, N_pc, pc_feature_dim]
        """
        # 1. 特征空间对齐
        aligned_img_features = self.feature_alignment(
            img_features, projection_matrix, voxel_coords
        )
        
        # 2. 跨模态注意力融合
        enhanced_features = self.cross_modal_attention(
            pc_features, aligned_img_features
        )
        
        return enhanced_features


# 工具函数
def create_voxel_coordinates(voxel_size: Tuple[int, int, int], 
                           device: torch.device) -> torch.Tensor:
    """
    创建体素坐标网格
    
    Args:
        voxel_size: 体素尺寸 (X, Y, Z)
        device: 设备
        
    Returns:
        coords: 体素坐标 [1, X*Y*Z, 3]
    """
    X, Y, Z = voxel_size
    x = torch.arange(X, device=device)
    y = torch.arange(Y, device=device) 
    z = torch.arange(Z, device=device)
    
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    coords = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)
    
    return coords.unsqueeze(0)  # [1, X*Y*Z, 3]


def compute_projection_matrix(intrinsics: torch.Tensor, 
                            extrinsics: torch.Tensor) -> torch.Tensor:
    """
    计算投影矩阵
    
    Args:
        intrinsics: 内参矩阵 [B, 3, 3]
        extrinsics: 外参矩阵 [B, 4, 4]
        
    Returns:
        projection: 投影矩阵 [B, 3, 4]
    """
    # 简化的投影矩阵计算
    # 实际实现需要根据具体的相机模型和体素化参数
    projection = torch.bmm(intrinsics, extrinsics[:, :3, :])
    return projection
