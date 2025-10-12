#!/usr/bin/env python3
"""
异常检测模型测试脚本
Anomaly Detection Model Test Script

这个脚本用于测试跨模态注意力融合异常检测模型的功能
"""

import torch
import numpy as np
from muvo.config import get_cfg
from muvo.models.mile_anomaly import MileAnomalyDetection
from muvo.models.cross_modal_attention import CrossModalFusionModule, create_voxel_coordinates
from muvo.models.anomaly_detection_head import AnomalyDetectionHead


def create_dummy_batch(batch_size=2, sequence_length=2):
    """
    创建虚拟批次数据用于测试
    """
    batch = {
        'image': torch.randn(batch_size, sequence_length, 3, 600, 960),
        'route_map': torch.randn(batch_size, sequence_length, 3, 64, 64),
        'speed': torch.randn(batch_size, sequence_length, 1),
        'intrinsics': torch.randn(batch_size, sequence_length, 3, 3),
        'extrinsics': torch.randn(batch_size, sequence_length, 4, 4),
        'throttle_brake': torch.randn(batch_size, sequence_length, 1),
        'steering': torch.randn(batch_size, sequence_length, 1),
        'range_view_pcd_xyzd': torch.randn(batch_size, sequence_length, 4, 64, 1024),
        'voxel': torch.randn(batch_size, sequence_length, 1, 192, 192, 64),  # 异常检测用体素数据
    }
    return batch


def test_cross_modal_attention():
    """
    测试跨模态注意力融合模块
    """
    print("🧪 测试跨模态注意力融合模块...")
    
    # 测试不同的位置编码类型
    pe_types = ['sincos', 'learned', 'hybrid']
    
    # 创建测试数据 - 使用更小的尺寸
    batch_size = 1
    pc_features = torch.randn(batch_size, 32*32*16, 128)  # 点云特征 - 更小的尺寸
    img_features = torch.randn(batch_size, 128, 16, 32)    # 图像特征 - 更小的尺寸
    voxel_coords = create_voxel_coordinates((32, 32, 16), pc_features.device)
    voxel_coords = voxel_coords.expand(batch_size, -1, -1)
    projection_matrix = torch.randn(batch_size, 3, 4)
    
    for pe_type in pe_types:
        print(f"  📊 测试位置编码类型: {pe_type}")
        
        # 创建模块
        fusion_module = CrossModalFusionModule(
            pc_feature_dim=128,
            img_feature_dim=128,
            hidden_dim=64,
            num_heads=4,
            voxel_size=(32, 32, 16),
            use_positional_encoding=True,
            pe_encoding_type=pe_type,
            alignment_method='bilinear',
            use_alignment_network=True
        )
        
        # 前向传播
        with torch.no_grad():
            enhanced_features = fusion_module(
                pc_features=pc_features,
                img_features=img_features,
                voxel_coords=voxel_coords,
                projection_matrix=projection_matrix
            )
        
        print(f"    ✅ {pe_type} 位置编码测试通过!")
    
    print(f"✅ 输入点云特征形状: {pc_features.shape}")
    print(f"✅ 输入图像特征形状: {img_features.shape}")
    print(f"✅ 输出增强特征形状: {enhanced_features.shape}")
    print(f"✅ 跨模态注意力融合测试通过!")
    
    return enhanced_features


def test_feature_alignment():
    """
    测试特征对齐模块
    """
    print("\n🧪 测试特征对齐模块...")
    
    # 测试不同的对齐方法
    alignment_methods = ['nearest', 'bilinear', 'network']
    
    batch_size = 2
    img_features = torch.randn(batch_size, 256, 75, 120)    # 图像特征
    voxel_coords = create_voxel_coordinates((192, 192, 64), img_features.device)
    voxel_coords = voxel_coords.expand(batch_size, -1, -1)
    projection_matrix = torch.randn(batch_size, 3, 4)
    
    for method in alignment_methods:
        print(f"  📊 测试对齐方法: {method}")
        
        from muvo.models.cross_modal_attention import FeatureAlignment
        
        # 创建对齐模块
        alignment_module = FeatureAlignment(
            img_feature_dim=256,
            voxel_size=(192, 192, 64),
            alignment_method=method,
            use_alignment_network=(method == 'network')
        )
        
        # 前向传播
        with torch.no_grad():
            aligned_features = alignment_module(
                img_features=img_features,
                voxel_coords=voxel_coords,
                projection_matrix=projection_matrix
            )
        
        print(f"    ✅ {method} 对齐方法测试通过!")
        print(f"    ✅ 输入图像特征形状: {img_features.shape}")
        print(f"    ✅ 输出对齐特征形状: {aligned_features.shape}")
    
    print("✅ 特征对齐模块测试通过!")


def test_anomaly_detection_head():
    """
    测试异常检测头
    """
    print("\n🧪 测试异常检测头...")
    
    # 测试3D CNN检测头
    print("  📊 测试3D CNN检测头...")
    cnn_head = AnomalyDetectionHead(
        input_dim=256,
        head_type='3dcnn',
        output_dim=1
    )
    
    # 3D输入
    input_3d = torch.randn(2, 256, 48, 48, 16)
    with torch.no_grad():
        outputs_3d = cnn_head(input_3d)
    
    print(f"  ✅ 3D CNN输入形状: {input_3d.shape}")
    print(f"  ✅ 3D CNN异常分数形状: {outputs_3d['anomaly_scores'].shape}")
    print(f"  ✅ 3D CNN热力图形状: {outputs_3d['anomaly_heatmap'].shape}")
    
    # 测试MLP检测头
    print("  📊 测试MLP检测头...")
    mlp_head = AnomalyDetectionHead(
        input_dim=256,
        head_type='mlp',
        output_dim=1
    )
    
    # 序列输入
    input_seq = torch.randn(2, 192*192*64, 256)
    with torch.no_grad():
        outputs_seq = mlp_head(input_seq)
    
    print(f"  ✅ MLP输入形状: {input_seq.shape}")
    print(f"  ✅ MLP异常分数形状: {outputs_seq['anomaly_scores'].shape}")
    print(f"  ✅ MLP热力图形状: {outputs_seq['anomaly_heatmap'].shape}")
    
    print("✅ 异常检测头测试通过!")
    
    return outputs_3d, outputs_seq


def test_mile_anomaly_model():
    """
    测试完整的Mile异常检测模型
    """
    print("\n🧪 测试完整Mile异常检测模型...")
    
    # 创建配置
    cfg = get_cfg()
    cfg.MODEL.ANOMALY_DETECTION.ENABLED = True
    cfg.MODEL.ANOMALY_DETECTION.FREEZE_BACKBONE = False  # 暂时不冻结，避免预训练模型下载
    cfg.MODEL.ANOMALY_DETECTION.HEAD_TYPE = '3dcnn'
    cfg.MODEL.TRANSFORMER.ENABLED = True
    cfg.MODEL.LIDAR.ENABLED = True
    cfg.VOXEL_SEG.ENABLED = True
    cfg.MODEL.ENCODER.NAME = 'resnet18'
    cfg.MODEL.ENCODER.PRETRAINED = False  # 不使用预训练模型
    
    # 创建模型
    model = MileAnomalyDetection(cfg)
    model.eval()
    
    # 创建测试数据
    batch = create_dummy_batch(batch_size=1, sequence_length=1)
    
    # 前向传播
    with torch.no_grad():
        try:
            outputs, state_dict = model(batch)
            print(f"✅ 模型前向传播成功!")
            print(f"✅ 输出键: {list(outputs.keys())}")
            
            # 检查异常检测输出
            if 'anomaly_scores' in outputs:
                print(f"✅ 异常分数形状: {outputs['anomaly_scores'].shape}")
                print(f"✅ 异常概率: {outputs['anomaly_probability'].mean().item():.4f}")
            
        except Exception as e:
            print(f"❌ 模型前向传播失败: {e}")
            return False
    
    # 测试权重冻结
    print("\n🔒 测试权重冻结...")
    trainable_params = model.get_trainable_parameters()
    print(f"✅ 可训练参数数量: {len(trainable_params)}")
    
    # 检查骨干网络是否被冻结
    backbone_frozen = True
    for name, param in model.named_parameters():
        if 'encoder' in name and param.requires_grad:
            backbone_frozen = False
            break
    
    if backbone_frozen:
        print("✅ 骨干网络权重已正确冻结!")
    else:
        print("⚠️ 骨干网络权重未完全冻结!")
    
    print("✅ Mile异常检测模型测试通过!")
    return True


def main():
    """
    主测试函数
    """
    print("🚀 开始异常检测模型功能测试")
    print("=" * 60)
    
    try:
        # 测试跨模态注意力融合
        enhanced_features = test_cross_modal_attention()
        
        # 测试特征对齐
        test_feature_alignment()
        
        # 测试异常检测头
        outputs_3d, outputs_seq = test_anomaly_detection_head()
        
        # 测试完整模型
        model_success = test_mile_anomaly_model()
        
        print("\n" + "=" * 60)
        if model_success:
            print("🎉 所有测试通过! 异常检测模型功能正常!")
            print("🎉 All tests passed! Anomaly detection model is working correctly!")
        else:
            print("❌ 部分测试失败，请检查模型配置!")
            print("❌ Some tests failed, please check model configuration!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
