#!/usr/bin/env python3
"""
异常检测训练脚本
Anomaly Detection Training Script

这个脚本演示如何使用跨模态注意力融合的异常检测模型进行训练
"""

import argparse
import torch
import lightning.pytorch as pl
from muvo.config import get_cfg
from muvo.models.mile_anomaly import MileAnomalyDetection
from muvo.data.dataset import DataModule
from muvo.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train anomaly detection model with cross-modal attention fusion')
    parser.add_argument('--config-file', default='muvo/configs/anomaly_detection.yml', 
                       help='path to config file')
    parser.add_argument('--gpus', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--max-epochs', type=int, default=100, help='maximum number of epochs')
    parser.add_argument('--resume-from-checkpoint', type=str, default=None,
                       help='resume training from checkpoint')
    parser.add_argument('opts', help='Modify config options using the command-line', 
                       default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    
    # 加载配置
    cfg = get_cfg(args)
    
    # 修改配置参数
    if args.batch_size:
        cfg.BATCHSIZE = args.batch_size
    if args.gpus:
        cfg.GPUS = args.gpus
    
    print("=" * 60)
    print("🚀 跨模态注意力融合异常检测训练")
    print("🚀 Cross-Modal Attention Fusion Anomaly Detection Training")
    print("=" * 60)
    print(f"配置文件: {args.config_file}")
    print(f"GPU数量: {cfg.GPUS}")
    print(f"批次大小: {cfg.BATCHSIZE}")
    print(f"最大轮数: {args.max_epochs}")
    print("=" * 60)
    
    # 创建数据模块
    print("📊 加载数据集...")
    data_module = DataModule(cfg)
    
    # 创建模型
    print("🏗️ 创建异常检测模型...")
    model = MileAnomalyDetection(cfg)
    
    # 冻结骨干网络权重
    if cfg.MODEL.ANOMALY_DETECTION.FREEZE_BACKBONE:
        print("🔒 冻结骨干网络权重...")
        model.freeze_backbone_weights()
        
        # 显示可训练参数
        trainable_params = model.get_trainable_parameters()
        print(f"✅ 可训练参数数量: {len(trainable_params)}")
        total_params = sum(p.numel() for p in trainable_params)
        print(f"✅ 可训练参数总数: {total_params:,}")
    
    # 创建训练器
    print("🎯 创建训练器...")
    trainer = Trainer(
        model=model,
        cfg=cfg,
        data_module=data_module,
        max_epochs=args.max_epochs,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    # 开始训练
    print("🚀 开始训练...")
    print("=" * 60)
    
    try:
        trainer.fit()
        print("✅ 训练完成!")
    except KeyboardInterrupt:
        print("⚠️ 训练被用户中断")
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        raise
    
    print("=" * 60)
    print("🎉 异常检测模型训练完成!")
    print("🎉 Anomaly Detection Model Training Completed!")


if __name__ == '__main__':
    main()
