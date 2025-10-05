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


class PositionalEncoding3D(nn.Module):
    """
    3D位置编码模块 - 解决注意力机制的置换不变性问题
    3D Positional Encoding Module - Solving Permutation Invariance in Attention
    
    支持三种位置编码方式：
    1. 固定3D Sincosoidal位置编码
    2. 可学习的位置编码
    3. 混合位置编码
    """
    
    def __init__(self, 
                 feature_dim: int = 64,
                 voxel_size: Tuple[int, int, int] = (192, 192, 64),
                 encoding_type: str = 'sincos',  # 'sincos', 'learned', 'hybrid'
                 max_len: int = 10000):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.voxel_size = voxel_size
        self.encoding_type = encoding_type
        self.max_len = max_len
        
        if encoding_type == 'sincos':
            self._create_sincos_encoding()
        elif encoding_type == 'learned':
            self._create_learned_encoding()
        elif encoding_type == 'hybrid':
            self._create_hybrid_encoding()
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    def _create_sincos_encoding(self):
        """创建固定的3D Sincosoidal位置编码"""
        X, Y, Z = self.voxel_size
        
        # 为每个维度创建位置编码
        pe_x = self._get_1d_encoding(X, 0)
        pe_y = self._get_1d_encoding(Y, 1) 
        pe_z = self._get_1d_encoding(Z, 2)
        
        # 广播到3D空间
        pe_x = pe_x.unsqueeze(1).unsqueeze(2).expand(-1, Y, Z)
        pe_y = pe_y.unsqueeze(0).unsqueeze(2).expand(X, -1, Z)
        pe_z = pe_z.unsqueeze(0).unsqueeze(1).expand(X, Y, -1)
        
        # 组合位置编码
        pe = pe_x + pe_y + pe_z
        pe = pe.unsqueeze(0).unsqueeze(0)  # [1, 1, X, Y, Z, feature_dim]
        
        self.register_buffer('pe', pe)
        
        # 位置编码投影层
        self.pe_projection = nn.Linear(self.feature_dim, self.feature_dim)
    
    def _get_1d_encoding(self, length: int, dim: int):
        """获取1D位置编码"""
        pe = torch.zeros(length, self.feature_dim)
        position = torch.arange(0, length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, self.feature_dim, 2).float() *
                           -(math.log(self.max_len) / self.feature_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def _create_learned_encoding(self):
        """创建可学习的位置编码"""
        X, Y, Z = self.voxel_size
        total_positions = X * Y * Z
        
        self.learned_pe = nn.Embedding(total_positions, self.feature_dim)
        
        # 初始化位置编码
        nn.init.normal_(self.learned_pe.weight, std=0.02)
    
    def _create_hybrid_encoding(self):
        """创建混合位置编码（固定+可学习）"""
        # 固定部分
        self._create_sincos_encoding()
        
        # 可学习部分
        X, Y, Z = self.voxel_size
        total_positions = X * Y * Z
        self.learned_pe = nn.Embedding(total_positions, self.feature_dim)
        
        # 混合权重
        self.mix_weight = nn.Parameter(torch.tensor(0.5))
        
        nn.init.normal_(self.learned_pe.weight, std=0.02)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码到特征
        
        Args:
            features: 输入特征 [B, C, X, Y, Z] 或 [B, N, C]
            
        Returns:
            features_with_pe: 带位置编码的特征
        """
        if self.encoding_type == 'sincos':
            return self._apply_sincos_encoding(features)
        elif self.encoding_type == 'learned':
            return self._apply_learned_encoding(features)
        elif self.encoding_type == 'hybrid':
            return self._apply_hybrid_encoding(features)
    
    def _apply_sincos_encoding(self, features: torch.Tensor) -> torch.Tensor:
        """应用固定位置编码"""
        if features.dim() == 5:  # [B, C, X, Y, Z]
            B, C, X, Y, Z = features.shape
            pe = self.pe.expand(B, C, X, Y, Z, -1)
            pe = pe.permute(0, 1, 2, 3, 4, 5).contiguous()
            pe = pe.view(B, C, X, Y, Z, -1)
            
            # 投影位置编码
            pe = self.pe_projection(pe)
            
            # 添加到特征
            features = features.unsqueeze(-1) + pe
            return features.squeeze(-1)
        else:  # [B, N, C]
            # 对于序列输入，需要重新构造3D位置编码
            B, N, C = features.shape
            X, Y, Z = self.voxel_size
            
            # 创建位置索引
            positions = torch.arange(N, device=features.device)
            x_pos = positions // (Y * Z)
            y_pos = (positions % (Y * Z)) // Z
            z_pos = positions % Z
            
            # 获取对应的位置编码
            pe = self.pe[0, 0, x_pos, y_pos, z_pos]  # [N, feature_dim]
            pe = pe.unsqueeze(0).expand(B, -1, -1)  # [B, N, feature_dim]
            
            # 投影并添加
            pe = self.pe_projection(pe)
            return features + pe
    
    def _apply_learned_encoding(self, features: torch.Tensor) -> torch.Tensor:
        """应用可学习位置编码"""
        if features.dim() == 5:  # [B, C, X, Y, Z]
            B, C, X, Y, Z = features.shape
            N = X * Y * Z
            
            # 创建位置索引
            positions = torch.arange(N, device=features.device)
            pe = self.learned_pe(positions)  # [N, feature_dim]
            pe = pe.view(X, Y, Z, -1).permute(3, 0, 1, 2)  # [feature_dim, X, Y, Z]
            pe = pe.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B, feature_dim, X, Y, Z]
            
            return features + pe
        else:  # [B, N, C]
            B, N, C = features.shape
            positions = torch.arange(N, device=features.device)
            pe = self.learned_pe(positions)  # [N, feature_dim]
            pe = pe.unsqueeze(0).expand(B, -1, -1)  # [B, N, feature_dim]
            
            return features + pe
    
    def _apply_hybrid_encoding(self, features: torch.Tensor) -> torch.Tensor:
        """应用混合位置编码"""
        sincos_pe = self._apply_sincos_encoding(features)
        learned_pe = self._apply_learned_encoding(features)
        
        # 混合两种编码
        mixed_pe = self.mix_weight * sincos_pe + (1 - self.mix_weight) * learned_pe
        return features + mixed_pe


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
                 dropout: float = 0.1,
                 use_positional_encoding: bool = True,
                 pe_encoding_type: str = 'sincos',
                 voxel_size: Tuple[int, int, int] = (192, 192, 64)):
        super().__init__()
        
        self.pc_feature_dim = pc_feature_dim
        self.img_feature_dim = img_feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_positional_encoding = use_positional_encoding
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # 位置编码模块
        if use_positional_encoding:
            self.pc_positional_encoding = PositionalEncoding3D(
                feature_dim=pc_feature_dim,
                voxel_size=voxel_size,
                encoding_type=pe_encoding_type
            )
            self.img_positional_encoding = PositionalEncoding3D(
                feature_dim=img_feature_dim,
                voxel_size=voxel_size,
                encoding_type=pe_encoding_type
            )
        
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
        
        # 添加位置编码
        if self.use_positional_encoding:
            pc_features = self.pc_positional_encoding(pc_features)
            img_features = self.img_positional_encoding(img_features)
        
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
    优化的特征空间对齐模块
    Enhanced Feature Space Alignment Module
    
    支持多种对齐方法：
    1. 最近邻插值 (Nearest Neighbor)
    2. 双线性插值 (Bilinear Interpolation) 
    3. 对齐网络 (Alignment Network)
    """
    
    def __init__(self, 
                 img_feature_dim: int = 64,
                 voxel_size: Tuple[int, int, int] = (192, 192, 64),
                 alignment_method: str = 'bilinear',  # 'nearest', 'bilinear', 'network'
                 use_alignment_network: bool = True):
        super().__init__()
        
        self.img_feature_dim = img_feature_dim
        self.voxel_size = voxel_size
        self.alignment_method = alignment_method
        self.use_alignment_network = use_alignment_network
        
        # 特征维度调整
        self.feature_adapter = nn.Conv2d(img_feature_dim, img_feature_dim, 1)
        
        # 对齐网络 - 学习更好的特征对齐
        if use_alignment_network:
            self.alignment_network = nn.Sequential(
                nn.Conv2d(img_feature_dim, img_feature_dim, 3, padding=1),
                nn.BatchNorm2d(img_feature_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(img_feature_dim, img_feature_dim, 3, padding=1),
                nn.BatchNorm2d(img_feature_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(img_feature_dim, img_feature_dim, 1)
            )
            
            # 注意力权重网络
            self.attention_weights = nn.Sequential(
                nn.Conv2d(img_feature_dim, img_feature_dim // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(img_feature_dim // 4, 1, 1),
                nn.Sigmoid()
            )
        
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
        
        # 对齐网络处理
        if self.use_alignment_network:
            img_features = self.alignment_network(img_features)
            # 应用注意力权重
            attention_weights = self.attention_weights(img_features)
            img_features = img_features * attention_weights
        
        # 根据对齐方法进行特征对齐
        if self.alignment_method == 'nearest':
            aligned_features = self._nearest_neighbor_alignment(
                img_features, voxel_coords, projection_matrix
            )
        elif self.alignment_method == 'bilinear':
            aligned_features = self._bilinear_alignment(
                img_features, voxel_coords, projection_matrix
            )
        elif self.alignment_method == 'network':
            aligned_features = self._network_alignment(
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
    
    def _bilinear_alignment(self, 
                           img_features: torch.Tensor,
                           voxel_coords: torch.Tensor, 
                           projection_matrix: torch.Tensor) -> torch.Tensor:
        """
        双线性插值对齐方法 - 更平滑的特征映射
        """
        B, C, H, W = img_features.shape
        N_voxel = voxel_coords.shape[1]
        
        # 将体素坐标投影到图像坐标（归一化到[-1, 1]）
        # 简化版本：直接映射
        x_coords = voxel_coords[:, :, 0] / (W - 1) * 2 - 1  # 归一化到[-1, 1]
        y_coords = voxel_coords[:, :, 1] / (H - 1) * 2 - 1
        
        # 创建网格用于双线性插值
        grid = torch.stack([x_coords, y_coords], dim=-1)  # [B, N_voxel, 2]
        grid = grid.unsqueeze(1)  # [B, 1, N_voxel, 2]
        
        # 双线性插值
        aligned_features = F.grid_sample(
            img_features, 
            grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )  # [B, C, 1, N_voxel]
        
        aligned_features = aligned_features.squeeze(2).permute(0, 2, 1)  # [B, N_voxel, C]
        
        return aligned_features
    
    def _network_alignment(self, 
                          img_features: torch.Tensor,
                          voxel_coords: torch.Tensor, 
                          projection_matrix: torch.Tensor) -> torch.Tensor:
        """
        网络对齐方法 - 使用学习到的对齐网络
        """
        B, C, H, W = img_features.shape
        N_voxel = voxel_coords.shape[1]
        
        # 首先使用双线性插值作为基础
        base_aligned = self._bilinear_alignment(img_features, voxel_coords, projection_matrix)
        
        # 然后通过网络进一步优化对齐
        # 这里可以添加更复杂的网络结构来学习对齐
        # 简化版本：使用线性变换
        if hasattr(self, 'alignment_network'):
            # 将对齐后的特征重新整形为空间格式进行网络处理
            X, Y, Z = self.voxel_size
            base_aligned_3d = base_aligned.view(B, C, X, Y, Z)
            
            # 应用对齐网络（需要适配3D输入）
            # 这里简化处理，实际可以设计专门的3D对齐网络
            enhanced_aligned = base_aligned_3d
            
            return enhanced_aligned.view(B, N_voxel, C)
        else:
            return base_aligned


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
                 dropout: float = 0.1,
                 use_positional_encoding: bool = True,
                 pe_encoding_type: str = 'sincos',
                 alignment_method: str = 'bilinear',
                 use_alignment_network: bool = True):
        super().__init__()
        
        # 特征对齐模块
        self.feature_alignment = FeatureAlignment(
            img_feature_dim=img_feature_dim,
            voxel_size=voxel_size,
            alignment_method=alignment_method,
            use_alignment_network=use_alignment_network
        )
        
        # 跨模态注意力模块
        self.cross_modal_attention = CrossModalAttention(
            pc_feature_dim=pc_feature_dim,
            img_feature_dim=img_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
            pe_encoding_type=pe_encoding_type,
            voxel_size=voxel_size
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
