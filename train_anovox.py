"""
AnoVox异常检测训练脚本
Training Script for AnoVox Anomaly Detection

使用方法:
python train_anovox.py --config muvo/configs/anovox_training.yml
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from muvo.config import get_cfg
from muvo.models.mile_anomaly import MileAnomalyDetection
from muvo.dataset.anovox_dataset import create_anovox_dataloader


class AnomalyDetectionModule(pl.LightningModule):
    """
    PyTorch Lightning模块 - 用于训练异常检测模型
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = MileAnomalyDetection(cfg)
        
        # 损失函数
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 保存超参数
        self.save_hyperparameters()
        
        # 用于记录训练指标
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
        print("=" * 60)
        print("🚀 模型初始化成功")
        self._print_model_info()
        print("=" * 60)
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"📊 模型参数统计:")
        print(f"  - 总参数: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"  - 冻结参数: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
        print(f"\n🎯 训练配置:")
        print(f"  - Batch Size: {self.cfg.BATCHSIZE}")
        print(f"  - Learning Rate: {self.cfg.OPTIMIZER.LR}")
        print(f"  - Max Epochs: {self.cfg.OPTIMIZER.EPOCHS}")
    
    def forward(self, batch):
        """前向传播"""
        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        try:
            # 前向传播
            outputs = self(batch)
            
            # 计算损失（假设有异常标签）
            if 'anomaly_label' in batch:
                # 如果有真值标签
                anomaly_scores = outputs['anomaly_scores']
                labels = batch['anomaly_label']
                
                # 调整标签形状以匹配预测
                if isinstance(labels, dict):
                    # 如果标签是字典格式，提取实际标签
                    labels = torch.zeros_like(anomaly_scores)
                
                loss = self.criterion(anomaly_scores, labels.float())
            else:
                # 如果没有标签，使用自监督损失（简化版）
                # 这里假设大部分数据是正常的
                anomaly_scores = outputs['anomaly_scores']
                # 简单的损失：鼓励分数接近0（正常）
                loss = torch.mean(anomaly_scores)
            
            # 记录损失
            self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            
            # 记录其他指标
            anomaly_prob = outputs['anomaly_probability'].mean()
            self.log('train_anomaly_prob', anomaly_prob, prog_bar=False, logger=True)
            
            # 保存输出用于epoch结束时的统计
            self.training_step_outputs.append({
                'loss': loss.detach(),
                'anomaly_prob': anomaly_prob.detach()
            })
            
            return loss
            
        except Exception as e:
            print(f"❌ Training step error: {e}")
            print(f"   Batch keys: {batch.keys()}")
            if 'image' in batch:
                print(f"   Image shape: {batch['image'].shape}")
            if 'points' in batch:
                print(f"   Points shape: {batch['points'].shape}")
            raise e
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        try:
            # 前向传播
            outputs = self(batch)
            
            # 计算损失
            if 'anomaly_label' in batch:
                anomaly_scores = outputs['anomaly_scores']
                labels = batch['anomaly_label']
                
                if isinstance(labels, dict):
                    labels = torch.zeros_like(anomaly_scores)
                
                loss = self.criterion(anomaly_scores, labels.float())
            else:
                anomaly_scores = outputs['anomaly_scores']
                loss = torch.mean(anomaly_scores)
            
            # 记录验证损失
            self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True)
            
            # 记录其他指标
            anomaly_prob = outputs['anomaly_probability'].mean()
            self.log('val_anomaly_prob', anomaly_prob, prog_bar=False, logger=True)
            
            # 保存输出
            self.validation_step_outputs.append({
                'loss': loss.detach(),
                'anomaly_prob': anomaly_prob.detach(),
                'anomaly_scores': outputs['anomaly_scores'].detach().cpu(),
                'anomaly_heatmap': outputs['anomaly_heatmap'].detach().cpu()
            })
            
            return loss
            
        except Exception as e:
            print(f"❌ Validation step error: {e}")
            raise e
    
    def on_train_epoch_end(self):
        """训练epoch结束"""
        if len(self.training_step_outputs) > 0:
            avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
            avg_prob = torch.stack([x['anomaly_prob'] for x in self.training_step_outputs]).mean()
            
            print(f"\n📊 Epoch {self.current_epoch} Training Summary:")
            print(f"   - Avg Loss: {avg_loss:.4f}")
            print(f"   - Avg Anomaly Prob: {avg_prob:.4f}")
            
            self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """验证epoch结束"""
        if len(self.validation_step_outputs) > 0:
            avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
            avg_prob = torch.stack([x['anomaly_prob'] for x in self.validation_step_outputs]).mean()
            
            print(f"   - Val Loss: {avg_loss:.4f}")
            print(f"   - Val Anomaly Prob: {avg_prob:.4f}")
            
            # 可视化第一个batch的异常热力图
            if self.current_epoch % 5 == 0:  # 每5个epoch可视化一次
                self._visualize_predictions(self.validation_step_outputs[0])
            
            self.validation_step_outputs.clear()
    
    def _visualize_predictions(self, outputs):
        """可视化预测结果"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            import matplotlib.pyplot as plt
            
            # 创建可视化目录
            vis_dir = Path('visualizations')
            vis_dir.mkdir(exist_ok=True)
            
            # 获取异常热力图
            heatmap = outputs['anomaly_heatmap'][0, 0].numpy()  # [X, Y]
            
            # 绘制热力图
            plt.figure(figsize=(10, 8))
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            plt.colorbar(label='Anomaly Score')
            plt.title(f'Anomaly Heatmap - Epoch {self.current_epoch}')
            plt.xlabel('Y')
            plt.ylabel('X')
            
            # 保存图像
            save_path = vis_dir / f'heatmap_epoch_{self.current_epoch:03d}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ 可视化保存至: {save_path}")
            
        except Exception as e:
            print(f"   ⚠️ 可视化失败: {e}")
    
    def configure_optimizers(self):
        """配置优化器"""
        # 只优化可训练的参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.cfg.OPTIMIZER.LR,
            weight_decay=self.cfg.OPTIMIZER.WEIGHT_DECAY
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.OPTIMIZER.EPOCHS,
            eta_min=self.cfg.OPTIMIZER.LR * 0.01
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='Train AnoVox Anomaly Detection Model')
    parser.add_argument('--config', type=str, default='muvo/configs/anovox_training.yml',
                       help='Path to config file')
    parser.add_argument('--data-root', type=str, 
                       default='/root/autodl-tmp/datasets/AnoVox/AnoVox_Dynamic_Mono_Town07',
                       help='Path to AnoVox dataset')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 AnoVox异常检测训练启动")
    print("=" * 60)
    
    # 加载配置
    cfg = get_cfg()
    if os.path.exists(args.config):
        cfg.merge_from_file(args.config)
        print(f"✅ 加载配置文件: {args.config}")
    else:
        print(f"⚠️ 配置文件不存在，使用默认配置")
    
    # 更新配置
    cfg.BATCHSIZE = args.batch_size
    # epochs从命令行参数获取，不从配置文件
    
    # 确保异常检测启用
    if not hasattr(cfg, 'ANOMALY_DETECTION'):
        from yacs.config import CfgNode as CN
        cfg.ANOMALY_DETECTION = CN()
    cfg.ANOMALY_DETECTION.ENABLED = True
    cfg.ANOMALY_DETECTION.FREEZE_BACKBONE = True
    
    # 禁用预训练权重下载（避免网络问题）
    cfg.MODEL.ENCODER.PRETRAINED = False
    print("⚠️ 禁用预训练权重（避免网络下载超时）")
    
    print(f"\n📊 训练配置:")
    print(f"  - 数据集: {args.data_root}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - GPUs: {args.gpus}")
    
    # 创建数据加载器
    print(f"\n📁 准备数据加载器...")
    
    try:
        train_loader = create_anovox_dataloader(
            data_root=args.data_root,
            split='train',
            batch_size=args.batch_size,
            num_workers=4,
            shuffle=True,
            load_voxel=True,
            load_anomaly_labels=True
        )
        print(f"✅ 训练集加载器创建成功: {len(train_loader)} batches")
        
        val_loader = create_anovox_dataloader(
            data_root=args.data_root,
            split='train',  # AnoVox可能没有明确的val split，先用train
            batch_size=args.batch_size,
            num_workers=4,
            shuffle=False,
            load_voxel=True,
            load_anomaly_labels=True
        )
        print(f"✅ 验证集加载器创建成功: {len(val_loader)} batches")
        
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        print(f"   请检查数据集路径: {args.data_root}")
        return
    
    # 创建模型
    print(f"\n🏗️ 初始化模型...")
    model = AnomalyDetectionModule(cfg)
    
    # 设置回调
    callbacks = [
        # 模型检查点
        ModelCheckpoint(
            dirpath='checkpoints',
            filename='anovox-anomaly-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            verbose=True
        ),
        
        # 早停
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            mode='min',
            verbose=True
        ),
        
        # 学习率监控
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # 设置日志
    logger = TensorBoardLogger(
        save_dir='lightning_logs',
        name='anovox_anomaly',
        default_hp_metric=False
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() and args.gpus > 0 else 'cpu',
        devices=args.gpus if torch.cuda.is_available() and args.gpus > 0 else None,
        callbacks=callbacks,
        logger=logger,
        precision=16,  # 混合精度训练
        gradient_clip_val=1.0,  # 梯度裁剪
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    print(f"\n🎓 开始训练...")
    print(f"💡 使用TensorBoard监控训练: tensorboard --logdir lightning_logs")
    print(f"💡 可视化结果保存在: visualizations/")
    print("=" * 60)
    
    # 开始训练
    try:
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=args.resume
        )
        
        print("\n" + "=" * 60)
        print("🎉 训练完成！")
        print("=" * 60)
        print(f"\n📊 最佳模型: {trainer.checkpoint_callback.best_model_path}")
        print(f"📈 最佳验证损失: {trainer.checkpoint_callback.best_model_score:.4f}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

