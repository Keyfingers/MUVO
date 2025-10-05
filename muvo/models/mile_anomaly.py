"""
修改后的Mile模型 - 集成跨模态注意力融合的异常检测
Modified Mile Model - Integrated Cross-Modal Attention Fusion for Anomaly Detection

这个模型实现了您方案中的完整架构：
1. Stage 1: 输入与预处理（数据预处理、点云体素化）
2. Stage 2: 冻结的骨干网络特征提取（图像和点云分支权重冻结）
3. Stage 3: 核心创新 - 跨模态注意力融合
4. Stage 4: 异常检测头与输出（仅此部分可训练）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Optional, Tuple

from constants import CARLA_FPS, DISPLAY_SEGMENTATION
from muvo.utils.network_utils import pack_sequence_dim, unpack_sequence_dim, remove_past
from muvo.models.common import BevDecoder, Decoder, RouteEncode, Policy, VoxelDecoder1, ConvDecoder, \
    PositionEmbeddingSine, DecoderDS, PointPillarNet, DownSampleConv
from muvo.models.frustum_pooling import FrustumPooling
from muvo.layers.layers import BasicBlock
from muvo.models.transition import RSSM
from muvo.models.cross_modal_attention import CrossModalFusionModule, create_voxel_coordinates, compute_projection_matrix
from muvo.models.anomaly_detection_head import AnomalyDetectionHead, FrozenBackboneWrapper, freeze_backbone_parameters


class MileAnomalyDetection(nn.Module):
    """
    基于Mile的异常检测模型 - 集成跨模态注意力融合
    
    核心创新：
    1. 冻结骨干网络权重，减少训练负担
    2. 跨模态注意力融合，增强特征表示
    3. 轻量级异常检测头，高效异常检测
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.receptive_field = cfg.RECEPTIVE_FIELD
        
        # 异常检测相关配置
        self.anomaly_detection_enabled = getattr(cfg, 'ANOMALY_DETECTION', {}).get('ENABLED', False)
        self.freeze_backbone = getattr(cfg, 'ANOMALY_DETECTION', {}).get('FREEZE_BACKBONE', True)
        
        embedding_n_channels = self.cfg.MODEL.EMBEDDING_DIM
        
        # ==================== Stage 2: 冻结的骨干网络特征提取 ====================
        
        # 图像特征编码器（权重冻结）
        if self.cfg.MODEL.ENCODER.NAME == 'resnet18':
            self.encoder = timm.create_model(
                cfg.MODEL.ENCODER.NAME, pretrained=True, features_only=True, out_indices=[2, 3, 4],
            )
            feature_info = self.encoder.feature_info.get_dicts(keys=['num_chs', 'reduction'])
            
            # 冻结图像编码器权重
            if self.freeze_backbone:
                self.encoder = FrozenBackboneWrapper(self.encoder, freeze=True)

        # 点云特征编码器（权重冻结）
        if self.cfg.MODEL.LIDAR.ENABLED:
            if self.cfg.MODEL.LIDAR.POINT_PILLAR.ENABLED:
                # Point-Pillar网络
                self.point_pillars = PointPillarNet(
                    num_input=8,
                    num_features=[32, 32],
                    min_x=-48,
                    max_x=48,
                    min_y=-48,
                    max_y=48,
                    pixels_per_meter=5)
                
                self.point_pillar_encoder = timm.create_model(
                    cfg.MODEL.LIDAR.ENCODER, pretrained=True, features_only=True, out_indices=[2, 3, 4], in_chans=32
                )
                point_pillar_feature_info = \
                    self.point_pillar_encoder.feature_info.get_dicts(keys=['num_chs', 'reduction'])
                self.point_pillar_decoder = DecoderDS(point_pillar_feature_info, self.cfg.MODEL.TRANSFORMER.CHANNELS)
                
                # 冻结点云编码器权重
                if self.freeze_backbone:
                    self.point_pillar_encoder = FrozenBackboneWrapper(self.point_pillar_encoder, freeze=True)
            else:
                # Range-view点云编码器
                self.range_view_encoder = timm.create_model(
                    cfg.MODEL.LIDAR.ENCODER, pretrained=True, features_only=True, out_indices=[2, 3, 4], in_chans=4
                )
                range_view_feature_info = self.range_view_encoder.feature_info.get_dicts(keys=['num_chs', 'reduction'])
                self.range_view_decoder = DecoderDS(range_view_feature_info, self.cfg.MODEL.TRANSFORMER.CHANNELS)
                
                # 冻结点云编码器权重
                if self.freeze_backbone:
                    self.range_view_encoder = FrozenBackboneWrapper(self.range_view_encoder, freeze=True)

        # 特征解码器
        if self.cfg.MODEL.TRANSFORMER.ENABLED:
            DecoderT = Decoder if self.cfg.MODEL.TRANSFORMER.LARGE else DecoderDS
            self.feat_decoder = DecoderT(feature_info, self.cfg.MODEL.TRANSFORMER.CHANNELS)
            
            # 图像特征压缩
            self.image_feature_conv = nn.Sequential(
                BasicBlock(self.cfg.MODEL.TRANSFORMER.CHANNELS, embedding_n_channels, stride=2, downsample=True),
                BasicBlock(embedding_n_channels, embedding_n_channels),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(start_dim=1),
            )
            
            # 点云特征压缩
            self.lidar_feature_conv = nn.Sequential(
                BasicBlock(self.cfg.MODEL.TRANSFORMER.CHANNELS, embedding_n_channels, stride=2, downsample=True),
                BasicBlock(embedding_n_channels, embedding_n_channels),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(start_dim=1),
            )
        else:
            self.feat_decoder = Decoder(feature_info, self.cfg.MODEL.ENCODER.OUT_CHANNELS)
            
        # ==================== Stage 3: 核心创新 - 跨模态注意力融合 ====================
        
        if self.anomaly_detection_enabled:
            # 跨模态融合模块
            self.cross_modal_fusion = CrossModalFusionModule(
                pc_feature_dim=self.cfg.MODEL.TRANSFORMER.CHANNELS,
                img_feature_dim=self.cfg.MODEL.TRANSFORMER.CHANNELS,
                hidden_dim=getattr(cfg.ANOMALY_DETECTION, 'HIDDEN_DIM', 128),
                num_heads=getattr(cfg.ANOMALY_DETECTION, 'NUM_HEADS', 8),
                voxel_size=cfg.VOXEL.SIZE,
                dropout=getattr(cfg.ANOMALY_DETECTION, 'DROPOUT', 0.1)
            )
            
            # ==================== Stage 4: 异常检测头 ====================
            
            # 异常检测头（仅此部分可训练）
            self.anomaly_detection_head = AnomalyDetectionHead(
                input_dim=self.cfg.MODEL.TRANSFORMER.CHANNELS,
                head_type=getattr(cfg.ANOMALY_DETECTION, 'HEAD_TYPE', '3dcnn'),
                output_dim=getattr(cfg.ANOMALY_DETECTION, 'OUTPUT_DIM', 1),
                dropout=getattr(cfg.ANOMALY_DETECTION, 'DROPOUT', 0.1)
            )
            
            # 体素特征提取器
            self.voxel_feature_extractor = nn.Sequential(
                nn.Conv3d(1, 32, kernel_size=3, padding=1),  # 假设体素输入为单通道
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, self.cfg.MODEL.TRANSFORMER.CHANNELS, kernel_size=3, padding=1),
                nn.BatchNorm3d(self.cfg.MODEL.TRANSFORMER.CHANNELS),
                nn.ReLU(inplace=True),
            )

        # ==================== 其他组件（保持原有功能） ====================
        
        # 位置编码
        if self.cfg.MODEL.TRANSFORMER.ENABLED:
            self.position_encode = PositionEmbeddingSine(
                num_pos_feats=self.cfg.MODEL.TRANSFORMER.CHANNELS // 2,
                normalize=True)
            
            # 传感器类型嵌入
            self.type_embedding = nn.Parameter(torch.zeros(1, 1, self.cfg.MODEL.TRANSFORMER.CHANNELS, 2))
            
            # Transformer编码器
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.cfg.MODEL.TRANSFORMER.CHANNELS,
                nhead=8,
                dropout=0.1,
            )
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        # 路由编码
        if self.cfg.MODEL.ROUTE.ENABLED:
            self.backbone_route = RouteEncode(self.cfg.MODEL.ROUTE.CHANNELS, cfg.MODEL.ROUTE.BACKBONE)

        # 速度编码
        self.speed_enc = nn.Sequential(
            nn.Linear(1, cfg.MODEL.SPEED.CHANNELS),
            nn.ReLU(True),
            nn.Linear(cfg.MODEL.SPEED.CHANNELS, cfg.MODEL.SPEED.CHANNELS),
            nn.ReLU(True),
        )
        self.speed_normalisation = cfg.SPEED.NORMALISATION

        # 特征融合
        feature_n_channels = 2 * embedding_n_channels
        if self.cfg.MODEL.ROUTE.ENABLED:
            feature_n_channels += self.cfg.MODEL.ROUTE.CHANNELS
        feature_n_channels += cfg.MODEL.SPEED.CHANNELS
        
        self.features_combine = nn.Linear(feature_n_channels, embedding_n_channels)

        # 循环模型
        if self.cfg.MODEL.TRANSITION.ENABLED:
            self.rssm = RSSM(
                embedding_dim=embedding_n_channels,
                action_dim=self.cfg.MODEL.ACTION_DIM,
                hidden_state_dim=self.cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM,
                state_dim=self.cfg.MODEL.TRANSITION.STATE_DIM,
                action_latent_dim=self.cfg.MODEL.TRANSITION.ACTION_LATENT_DIM,
                receptive_field=self.receptive_field,
                use_dropout=self.cfg.MODEL.TRANSITION.USE_DROPOUT,
                dropout_probability=self.cfg.MODEL.TRANSITION.DROPOUT_PROBABILITY,
            )

        # 策略网络
        if self.cfg.MODEL.TRANSITION.ENABLED:
            state_dim = self.cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM + self.cfg.MODEL.TRANSITION.STATE_DIM
        else:
            state_dim = embedding_n_channels
        self.policy = Policy(in_channels=state_dim)

        # 其他解码器（保持原有功能）
        if self.cfg.SEMANTIC_SEG.ENABLED:
            self.bev_decoder = BevDecoder(
                latent_n_channels=state_dim,
                semantic_n_channels=self.cfg.SEMANTIC_SEG.N_CHANNELS,
                head='bev',
            )

        if self.cfg.VOXEL_SEG.ENABLED:
            self.voxel_decoder = VoxelDecoder1(
                latent_n_channels=state_dim,
                semantic_n_channels=self.cfg.VOXEL_SEG.N_CLASSES,
                feature_channels=self.cfg.VOXEL_SEG.DIMENSION,
                constant_size=(3, 3, 1),
            )

        # 部署时保存状态
        self.last_h = None
        self.last_sample = None
        self.last_action = None
        self.count = 0

    def forward(self, batch, deployment=False):
        """
        前向传播
        
        Parameters
        ----------
            batch: dict of torch.Tensor
                keys:
                    image: (b, s, 3, h, w)
                    route_map: (b, s, 3, h_r, w_r)
                    speed: (b, s, 1)
                    intrinsics: (b, s, 3, 3)
                    extrinsics: (b, s, 4, 4)
                    throttle_brake: (b, s, 1)
                    steering: (b, s, 1)
                    voxel: (b, s, 1, x, y, z) - 体素数据（异常检测用）
        """
        # 编码输入到嵌入向量
        embedding = self.encode(batch)
        b, s = batch['image'].shape[:2]

        output = dict()
        
        # 异常检测分支
        if self.anomaly_detection_enabled and 'voxel' in batch:
            anomaly_output = self.anomaly_detection_forward(batch)
            output.update(anomaly_output)

        # 原有的循环模型和策略网络
        if self.cfg.MODEL.TRANSITION.ENABLED:
            if deployment:
                action = batch['action']
            else:
                action = torch.cat([batch['throttle_brake'], batch['steering']], dim=-1)
            state_dict = self.rssm(embedding, action, use_sample=not deployment, policy=self.policy)

            if deployment:
                state_dict = remove_past(state_dict, s)
                s = 1

            output = {**output, **state_dict}
            state = torch.cat([state_dict['posterior']['hidden_state'], state_dict['posterior']['sample']], dim=-1)
        else:
            state = embedding
            state_dict = {}

        # 策略输出
        state = pack_sequence_dim(state)
        output_policy = self.policy(state)
        throttle_brake, steering = torch.split(output_policy, 1, dim=-1)
        output['throttle_brake'] = unpack_sequence_dim(throttle_brake, b, s)
        output['steering'] = unpack_sequence_dim(steering, b, s)

        # 其他解码器输出
        if self.cfg.SEMANTIC_SEG.ENABLED:
            if (not deployment) or (deployment and DISPLAY_SEGMENTATION):
                bev_decoder_output = self.bev_decoder(state)
                bev_decoder_output = unpack_sequence_dim(bev_decoder_output, b, s)
                output = {**output, **bev_decoder_output}

        if self.cfg.VOXEL_SEG.ENABLED:
            voxel_decoder_output = self.voxel_decoder(state)
            voxel_decoder_output = unpack_sequence_dim(voxel_decoder_output, b, s)
            output = {**output, **voxel_decoder_output}

        return output, state_dict

    def anomaly_detection_forward(self, batch):
        """
        异常检测前向传播
        
        实现完整的跨模态注意力融合流程
        """
        b, s = batch['image'].shape[:2]
        
        # 打包序列维度
        image = pack_sequence_dim(batch['image'])
        voxel = pack_sequence_dim(batch['voxel'])
        intrinsics = pack_sequence_dim(batch['intrinsics'])
        extrinsics = pack_sequence_dim(batch['extrinsics'])
        
        # ==================== Stage 1: 数据预处理 ====================
        
        # 图像特征提取（冻结权重）
        img_features = self.encoder(image)
        img_features = self.feat_decoder(img_features)  # [B, C, H, W]
        
        # 点云特征提取（冻结权重）
        if self.cfg.MODEL.LIDAR.ENABLED:
            if self.cfg.MODEL.LIDAR.POINT_PILLAR.ENABLED:
                lidar_list = pack_sequence_dim(batch['points_raw'])
                num_points = pack_sequence_dim(batch['num_points'])
                pp_features = self.point_pillars(lidar_list, num_points)
                pp_xs = self.point_pillar_encoder(pp_features)
                pc_features = self.point_pillar_decoder(pp_xs)
            else:
                range_view = pack_sequence_dim(batch['range_view_pcd_xyzd'])
                lidar_xs = self.range_view_encoder(range_view)
                pc_features = self.range_view_decoder(lidar_xs)
        else:
            # 如果没有点云数据，使用体素特征
            pc_features = self.voxel_feature_extractor(voxel)
        
        # ==================== Stage 2: 特征空间对齐 ====================
        
        # 创建体素坐标
        voxel_coords = create_voxel_coordinates(self.cfg.VOXEL.SIZE, image.device)
        voxel_coords = voxel_coords.expand(b, -1, -1)  # [B, N_voxel, 3]
        
        # 计算投影矩阵
        projection_matrix = compute_projection_matrix(intrinsics, extrinsics)
        
        # ==================== Stage 3: 跨模态注意力融合 ====================
        
        # 将特征重塑为序列格式
        B, C, H, W = img_features.shape
        img_features_seq = img_features.view(B, C, H*W).permute(0, 2, 1)  # [B, H*W, C]
        
        B, C, X, Y, Z = pc_features.shape
        pc_features_seq = pc_features.view(B, C, X*Y*Z).permute(0, 2, 1)  # [B, X*Y*Z, C]
        
        # 跨模态注意力融合
        enhanced_features = self.cross_modal_fusion(
            pc_features=pc_features_seq,
            img_features=img_features,
            voxel_coords=voxel_coords,
            projection_matrix=projection_matrix
        )
        
        # 重塑回3D格式
        enhanced_features_3d = enhanced_features.permute(0, 2, 1).view(B, C, X, Y, Z)
        
        # ==================== Stage 4: 异常检测 ====================
        
        # 异常检测头
        anomaly_outputs = self.anomaly_detection_head(enhanced_features_3d)
        
        # 解包序列维度
        for key, value in anomaly_outputs.items():
            anomaly_outputs[key] = unpack_sequence_dim(value, b, s)
        
        return anomaly_outputs

    def encode(self, batch):
        """
        编码输入数据（保持原有功能）
        """
        b, s = batch['image'].shape[:2]
        image = pack_sequence_dim(batch['image'])
        speed = pack_sequence_dim(batch['speed'])
        intrinsics = pack_sequence_dim(batch['intrinsics'])
        extrinsics = pack_sequence_dim(batch['extrinsics'])

        # 图像编码
        xs = self.encoder(image)
        x = self.feat_decoder(xs)

        if self.cfg.MODEL.TRANSFORMER.ENABLED:
            # 获取点云特征
            if self.cfg.MODEL.LIDAR.ENABLED:
                if self.cfg.MODEL.LIDAR.POINT_PILLAR.ENABLED:
                    lidar_list = pack_sequence_dim(batch['points_raw'])
                    num_points = pack_sequence_dim(batch['num_points'])
                    pp_features = self.point_pillars(lidar_list, num_points)
                    pp_xs = self.point_pillar_encoder(pp_features)
                    lidar_features = self.point_pillar_decoder(pp_xs)
                else:
                    range_view = pack_sequence_dim(batch['range_view_pcd_xyzd'])
                    lidar_xs = self.range_view_encoder(range_view)
                    lidar_features = self.range_view_decoder(lidar_xs)
                
                bs_image, _, h_image, w_image = x.shape
                bs_lidar, _, h_lidar, w_lidar = lidar_features.shape

                # 位置编码
                image_tokens = x + self.position_encode(x)
                lidar_tokens = lidar_features + self.position_encode(lidar_features)

                # 展平特征
                image_tokens = image_tokens.flatten(start_dim=2).permute(2, 0, 1)
                lidar_tokens = lidar_tokens.flatten(start_dim=2).permute(2, 0, 1)

                # 传感器类型嵌入
                image_tokens += self.type_embedding[:, :, :, 0]
                lidar_tokens += self.type_embedding[:, :, :, 1]

                L_image, _, _ = image_tokens.shape
                L_lidar, _, _ = lidar_tokens.shape

                # 连接图像和点云tokens
                tokens = torch.cat([image_tokens, lidar_tokens], dim=0)
                tokens_out = self.transformer_encoder(tokens)
                
                # 分离并重塑
                image_tokens_out = tokens_out[:L_image].permute(1, 2, 0).reshape((bs_image, -1, h_image, w_image))
                lidar_tokens_out = tokens_out[L_image:].permute(1, 2, 0).reshape((bs_lidar, -1, h_lidar, w_lidar))

                # 压缩到1D
                image_features_out = self.image_feature_conv(image_tokens_out)
                lidar_features_out = self.lidar_feature_conv(lidar_tokens_out)

                features = [image_features_out, lidar_features_out]
            else:
                # 仅图像特征
                image_features_out = self.image_feature_conv(x)
                features = [image_features_out]

            # 其他特征
            if self.cfg.MODEL.ROUTE.ENABLED:
                route_map = pack_sequence_dim(batch['route_map'])
                route_map_features = self.backbone_route(route_map)
                features.append(route_map_features)

            speed_features = self.speed_enc(speed / self.speed_normalisation)
            features.append(speed_features)

            embedding = self.features_combine(torch.cat(features, dim=-1))

        embedding = unpack_sequence_dim(embedding, b, s)
        return embedding

    def get_trainable_parameters(self):
        """
        获取可训练参数（仅异常检测相关模块）
        """
        trainable_params = []
        
        if self.anomaly_detection_enabled:
            # 跨模态融合模块参数
            trainable_params.extend(self.cross_modal_fusion.parameters())
            
            # 异常检测头参数
            trainable_params.extend(self.anomaly_detection_head.parameters())
            
            # 体素特征提取器参数
            trainable_params.extend(self.voxel_feature_extractor.parameters())
        
        return trainable_params

    def freeze_backbone_weights(self):
        """
        冻结骨干网络权重
        """
        if self.freeze_backbone:
            freeze_backbone_parameters(self, ['encoder', 'range_view_encoder', 'point_pillar_encoder'])
            print("Backbone weights frozen for anomaly detection training")
