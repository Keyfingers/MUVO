"""
异常检测头模块 - 可训练的异常检测网络
Anomaly Detection Head Module - Trainable Anomaly Detection Network

这个模块实现了您方案中的Stage 4：异常检测头与输出
- 轻量级3D CNN或MLP架构
- 仅此部分可训练，其他骨干网络权重冻结
- 输出体素级异常分数和异常热力图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class Lightweight3DCNN(nn.Module):
    """
    轻量级3D CNN异常检测头
    Lightweight 3D CNN Anomaly Detection Head
    """
    
    def __init__(self,
                 input_dim: int = 64,
                 hidden_dims: Tuple[int, ...] = (128, 64, 32),
                 output_dim: int = 1,  # 异常分数
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 构建3D CNN层
        layers = []
        in_channels = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout3d(dropout)
            ])
            in_channels = hidden_dim
            
        # 输出层
        layers.append(nn.Conv3d(in_channels, output_dim, kernel_size=1))
        
        self.cnn_layers = nn.Sequential(*layers)
        
        # 激活函数
        self.activation = nn.Sigmoid()  # 输出0-1之间的异常分数
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, C, X, Y, Z]
            
        Returns:
            anomaly_scores: 异常分数 [B, 1, X, Y, Z]
        """
        anomaly_scores = self.cnn_layers(x)
        anomaly_scores = self.activation(anomaly_scores)
        
        return anomaly_scores


class MLPAnomalyHead(nn.Module):
    """
    MLP异常检测头
    MLP Anomaly Detection Head
    """
    
    def __init__(self,
                 input_dim: int = 64,
                 hidden_dims: Tuple[int, ...] = (256, 128, 64),
                 output_dim: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 构建MLP层
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
            
        # 输出层
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.mlp_layers = nn.Sequential(*layers)
        
        # 激活函数
        self.activation = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, N, C] 或 [B, C, X, Y, Z]
            
        Returns:
            anomaly_scores: 异常分数 [B, N, 1] 或 [B, 1, X, Y, Z]
        """
        # 处理不同的输入形状
        if x.dim() == 4:  # [B, C, X, Y, Z]
            B, C, X, Y, Z = x.shape
            x = x.permute(0, 2, 3, 4, 1).contiguous().view(B * X * Y * Z, C)
            is_3d = True
        else:  # [B, N, C]
            B, N, C = x.shape
            x = x.view(B * N, C)
            is_3d = False
            
        # MLP前向传播
        anomaly_scores = self.mlp_layers(x)
        anomaly_scores = self.activation(anomaly_scores)
        
        # 恢复原始形状
        if is_3d:
            anomaly_scores = anomaly_scores.view(B, X, Y, Z, self.output_dim)
            anomaly_scores = anomaly_scores.permute(0, 4, 1, 2, 3).contiguous()
        else:
            anomaly_scores = anomaly_scores.view(B, N, self.output_dim)
            
        return anomaly_scores


class MultiScaleAnomalyHead(nn.Module):
    """
    多尺度异常检测头
    Multi-Scale Anomaly Detection Head
    """
    
    def __init__(self,
                 input_dim: int = 64,
                 scales: Tuple[int, ...] = (1, 2, 4),
                 output_dim: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        
        self.scales = scales
        self.output_dim = output_dim
        
        # 为每个尺度创建检测头
        self.scale_heads = nn.ModuleList([
            Lightweight3DCNN(
                input_dim=input_dim,
                hidden_dims=(128, 64, 32),
                output_dim=output_dim,
                dropout=dropout
            ) for _ in scales
        ])
        
        # 特征融合
        self.fusion_conv = nn.Conv3d(
            len(scales) * output_dim, 
            output_dim, 
            kernel_size=1
        )
        
        # 最终激活
        self.activation = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        多尺度异常检测
        
        Args:
            x: 输入特征 [B, C, X, Y, Z]
            
        Returns:
            anomaly_scores: 异常分数 [B, 1, X, Y, Z]
        """
        scale_outputs = []
        
        for i, scale in enumerate(self.scales):
            # 下采样
            if scale > 1:
                scaled_x = F.avg_pool3d(x, kernel_size=scale, stride=scale)
            else:
                scaled_x = x
                
            # 通过对应的检测头
            scale_output = self.scale_heads[i](scaled_x)
            
            # 上采样到原始尺寸
            if scale > 1:
                scale_output = F.interpolate(
                    scale_output, 
                    size=x.shape[2:], 
                    mode='trilinear', 
                    align_corners=False
                )
                
            scale_outputs.append(scale_output)
            
        # 融合多尺度输出
        fused_output = torch.cat(scale_outputs, dim=1)
        anomaly_scores = self.fusion_conv(fused_output)
        anomaly_scores = self.activation(anomaly_scores)
        
        return anomaly_scores


class AnomalyDetectionHead(nn.Module):
    """
    异常检测头主模块
    Main Anomaly Detection Head Module
    """
    
    def __init__(self,
                 input_dim: int = 64,
                 head_type: str = '3dcnn',  # '3dcnn', 'mlp', 'multiscale'
                 output_dim: int = 1,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        
        self.head_type = head_type
        self.output_dim = output_dim
        
        # 根据类型选择检测头
        if head_type == '3dcnn':
            self.detection_head = Lightweight3DCNN(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=dropout,
                **kwargs
            )
        elif head_type == 'mlp':
            self.detection_head = MLPAnomalyHead(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=dropout,
                **kwargs
            )
        elif head_type == 'multiscale':
            self.detection_head = MultiScaleAnomalyHead(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=dropout,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown head type: {head_type}")
            
    def forward(self, enhanced_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        异常检测前向传播
        
        Args:
            enhanced_features: 增强的特征 [B, C, X, Y, Z] 或 [B, N, C]
            
        Returns:
            outputs: 包含异常分数和热力图的字典
        """
        # 异常分数预测
        anomaly_scores = self.detection_head(enhanced_features)
        
        # 生成异常热力图
        if anomaly_scores.dim() == 5:  # 3D体素
            anomaly_heatmap = self._generate_3d_heatmap(anomaly_scores)
        else:  # 2D或1D
            anomaly_heatmap = self._generate_2d_heatmap(anomaly_scores)
            
        outputs = {
            'anomaly_scores': anomaly_scores,
            'anomaly_heatmap': anomaly_heatmap,
            'anomaly_probability': torch.mean(anomaly_scores, dim=[2, 3, 4] if anomaly_scores.dim() == 5 else [1, 2])
        }
        
        return outputs
        
    def _generate_3d_heatmap(self, anomaly_scores: torch.Tensor) -> torch.Tensor:
        """
        生成3D异常热力图
        """
        # 对Z轴进行最大池化，生成2D热力图
        heatmap_2d = torch.max(anomaly_scores, dim=4)[0]  # [B, 1, X, Y]
        
        # 归一化到0-1
        heatmap_2d = (heatmap_2d - heatmap_2d.min()) / (heatmap_2d.max() - heatmap_2d.min() + 1e-8)
        
        return heatmap_2d
        
    def _generate_2d_heatmap(self, anomaly_scores: torch.Tensor) -> torch.Tensor:
        """
        生成2D异常热力图
        """
        if anomaly_scores.dim() == 4:  # [B, 1, H, W]
            heatmap = anomaly_scores
        else:  # [B, N, 1]
            # 重塑为2D
            B, N, _ = anomaly_scores.shape
            H = W = int(N ** 0.5)
            heatmap = anomaly_scores.view(B, 1, H, W)
            
        # 归一化
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap


class FrozenBackboneWrapper(nn.Module):
    """
    冻结骨干网络包装器
    Frozen Backbone Wrapper
    
    用于冻结图像和点云骨干网络的权重
    """
    
    def __init__(self, backbone: nn.Module, freeze: bool = True):
        super().__init__()
        self.backbone = backbone
        self.freeze = freeze
        
        if freeze:
            self._freeze_backbone()
            
    def _freeze_backbone(self):
        """冻结骨干网络参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def forward(self, *args, **kwargs):
        """前向传播（不计算梯度）"""
        if self.freeze:
            with torch.no_grad():
                return self.backbone(*args, **kwargs)
        else:
            return self.backbone(*args, **kwargs)


# 工具函数
def create_anomaly_detection_head(config: Dict) -> AnomalyDetectionHead:
    """
    根据配置创建异常检测头
    
    Args:
        config: 配置字典
        
    Returns:
        anomaly_head: 异常检测头实例
    """
    return AnomalyDetectionHead(**config)


def freeze_backbone_parameters(model: nn.Module, 
                              backbone_names: list = ['encoder', 'range_view_encoder']):
    """
    冻结骨干网络参数
    
    Args:
        model: 模型实例
        backbone_names: 需要冻结的骨干网络名称列表
    """
    for name, module in model.named_modules():
        if any(backbone_name in name for backbone_name in backbone_names):
            for param in module.parameters():
                param.requires_grad = False
            print(f"Frozen parameters in: {name}")


def get_trainable_parameters(model: nn.Module) -> list:
    """
    获取可训练参数列表
    
    Args:
        model: 模型实例
        
    Returns:
        trainable_params: 可训练参数列表
    """
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param))
    return trainable_params
