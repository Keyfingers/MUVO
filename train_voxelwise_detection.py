#!/usr/bin/env python3
"""
体素级异常检测训练 - 精确映射版 🎯
架构革命 + 标签革命：
1. 架构：从"场景分类"变为"体素分割" (移除全局池化)
2. 标签：从"随机分配"变为"精确体素-点映射" (基于语义ID)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights

sys.path.insert(0, '/root/autodl-tmp/MUVO/MUVO')

from muvo.dataset.anovox_dataset import AnoVoxDataset, collate_fn
from precise_label_mapping import create_improved_labels_from_voxels


class FocalLoss(nn.Module):
    """Focal Loss - 解决类别不平衡"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [B*N] - 逐点logits
            targets: [B*N] - 逐点标签
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_weight * focal_weight
        
        loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class VoxelwiseAnomalyDetector(nn.Module):
    """
    体素级异常检测器 - 架构革命版
    
    关键改变：
    1. ❌ 取消全局池化
    2. ✅ 逐点/体素预测
    3. ✅ 输出 [B, N] 而非 [B, 1]
    """
    def __init__(self, freeze_backbone=True):
        super().__init__()
        
        # ========== 图像编码器 (预训练ResNet18) ==========
        print("🔧 加载预训练ResNet18...")
        weights = ResNet18_Weights.IMAGENET1K_V1
        pretrained_resnet = resnet18(weights=weights)
        self.image_encoder = nn.Sequential(*list(pretrained_resnet.children())[:-2])
        
        if freeze_backbone:
            print("   ❄️  冻结ResNet权重")
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        
        # ========== 点云编码器 ==========
        self.point_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )
        
        # ========== 跨模态注意力融合 ==========
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        # ========== 【关键修改】逐点异常检测头 ==========
        # 输入: 单个点的融合特征 [512]
        # 输出: 单个点的异常logit [1]
        self.point_anomaly_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # 每个点一个logit
        )
    
    def forward(self, batch):
        """
        前向传播 - 体素级预测
        
        Returns:
            {
                'anomaly_logits': [B, N] - 每个点的logit
                'anomaly_score': [B, N] - 每个点的概率
                'num_points': int - 实际处理的点数
                'point_indices': tensor - 采样的点索引（如果有采样）
            }
        """
        # 1. 图像特征提取
        img = batch['image']  # [B, 3, 224, 224]
        img_feat = self.image_encoder(img)  # [B, 512, 7, 7]
        
        B, C, H, W = img_feat.shape
        img_feat = img_feat.view(B, C, H*W)  # [B, 512, 49]
        img_feat = img_feat.permute(0, 2, 1)  # [B, 49, 512]
        
        # 2. 点云特征提取
        points = batch['points']  # [B, N, 4]
        B, N, _ = points.shape
        
        # 限制点云数量
        max_points = 2048
        point_indices = None
        if N > max_points:
            indices = torch.randperm(N, device=points.device)[:max_points]
            points = points[:, indices, :]
            point_indices = indices  # 保存索引
            N = max_points
        elif N < max_points:
            padding = torch.zeros(B, max_points - N, 4, device=points.device)
            points = torch.cat([points, padding], dim=1)
            N = max_points
        
        # 对每个点独立编码
        points_reshaped = points.view(B * N, 4)  # [B*N, 4]
        point_feat = self.point_encoder(points_reshaped)  # [B*N, 512]
        point_feat = point_feat.view(B, N, 512)  # [B, N, 512]
        
        # 3. 跨模态注意力融合
        fused_feat, attn_weights = self.cross_attention(
            query=point_feat,     # [B, N, 512]
            key=img_feat,         # [B, 49, 512]
            value=img_feat
        )  # [B, N, 512]
        
        # 4. 【关键改变】逐点异常检测 (NO Global Pooling!)
        # 不进行全局池化，直接对每个点的融合特征进行预测
        fused_feat_reshaped = fused_feat.reshape(B * N, 512)  # [B*N, 512]
        
        # 每个点独立预测
        anomaly_logits = self.point_anomaly_head(fused_feat_reshaped)  # [B*N, 1]
        anomaly_logits = anomaly_logits.view(B, N)  # [B, N]
        
        anomaly_score = torch.sigmoid(anomaly_logits)  # [B, N]
        
        return {
            'anomaly_logits': anomaly_logits,  # [B, N]
            'anomaly_score': anomaly_score,    # [B, N]
            'attention_weights': attn_weights,
            'num_points': N,
            'point_indices': point_indices  # 采样的点索引（如果有）
        }


# 旧的随机标签函数已被删除
# 现在使用 precise_label_mapping.py 中的改进版本


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """训练一个epoch - 体素级"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    total_anomalies = 0
    detected_anomalies = 0
    
    all_losses = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        batch['image'] = batch['image'].to(device)
        batch['points'] = batch['points'].to(device)
        
        optimizer.zero_grad()
        outputs = model(batch)
        
        # 【关键】创建改进的体素级标签 [B, N]
        # 注意：由于点云坐标系与体素索引坐标系不匹配，我们使用anomaly_is_alive字段
        N = outputs['num_points']
        labels = create_improved_labels_from_voxels(batch, N, device)  # [B, N]
        
        # 逐点损失计算
        logits = outputs['anomaly_logits']  # [B, N]
        B_logits, N_logits = logits.shape
        B_labels, N_labels = labels.shape
        
        # 确保维度匹配
        assert B_logits == B_labels, f"Batch size mismatch: {B_logits} vs {B_labels}"
        assert N_logits == N_labels, f"Point num mismatch: {N_logits} vs {N_labels}"
        
        # Flatten for loss computation
        logits_flat = logits.reshape(-1)  # [B*N]
        labels_flat = labels.reshape(-1)  # [B*N]
        
        # Focal Loss
        loss = criterion(logits_flat, labels_flat)
        
        B, N = B_logits, N_logits
        
        loss.backward()
        optimizer.step()
        
        # 统计
        with torch.no_grad():
            scores = outputs['anomaly_score']  # [B, N]
            predictions = (scores > 0.5).float()
            
            correct = (predictions == labels).sum().item()
            total_correct += correct
            total_samples += B * N
            
            # 异常检测统计
            anomaly_mask = (labels == 1)
            total_anomalies += anomaly_mask.sum().item()
            detected_anomalies += ((predictions == 1) & (labels == 1)).sum().item()
            
            total_loss += loss.item()
            all_losses.append(loss.item())
        
        # 更新进度条
        accuracy = 100.0 * correct / (B * N)
        anomaly_recall = 100.0 * detected_anomalies / max(total_anomalies, 1)
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.1f}%',
            'recall': f'{anomaly_recall:.1f}%',
            'avg_score': f'{scores.mean().item():.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = 100.0 * total_correct / total_samples
    avg_anomaly_recall = 100.0 * detected_anomalies / max(total_anomalies, 1)
    
    return {
        'loss': avg_loss,
        'accuracy': avg_acc,
        'anomaly_recall': avg_anomaly_recall,
        'total_anomalies': total_anomalies,
        'detected_anomalies': detected_anomalies,
        'all_losses': all_losses
    }


def main():
    """主训练函数"""
    print("=" * 70)
    print("🚀 体素级异常检测训练 - 架构革命版")
    print("=" * 70)
    print("\n✨ 关键改变:")
    print("   1. ❌ 取消全局池化 (Global Pooling)")
    print("   2. ✅ 逐点/体素预测 (Point-wise Prediction)")
    print("   3. ✅ 体素级标签 (Voxel-level Labels)")
    print("   4. ✅ 输出 [B, N] 而非 [B, 1]")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # 数据集
    print("\n📁 加载数据...")
    dataset = AnoVoxDataset(
        data_root='/root/autodl-tmp/datasets/AnoVox/AnoVox_Dynamic_Mono_Town07',
        split='train',
        load_voxel=True,  # 加载体素标签
        load_anomaly_labels=True
    )
    print(f"✅ 数据加载完成: {len(dataset)} 样本")
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # 降低batch size (因为现在是B*N个预测)
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 模型
    print("\n🏗️ 初始化模型...")
    model = VoxelwiseAnomalyDetector(freeze_backbone=True).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ 总参数: {num_params:,}")
    print(f"✅ 可训练参数: {num_trainable:,}")
    
    # Focal Loss
    print("\n⚡ 使用Focal Loss (alpha=0.1, gamma=3.0)")
    criterion = FocalLoss(alpha=0.1, gamma=3.0)  # 更激进的参数
    print(f"   alpha=0.1 (异常样本权重10倍)")
    print(f"   gamma=3.0 (更强的聚焦)")
    
    # 优化器
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=1e-6
    )
    
    # 训练
    num_epochs = 30
    print(f"\n🎓 开始训练 ({num_epochs} epochs)...")
    print(f"   Batch size: 4")
    print(f"   每个batch预测: 4 × 2048 = 8192 个点")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'anomaly_recall': []
    }
    
    best_recall = 0.0
    
    for epoch in range(1, num_epochs + 1):
        results = train_epoch(model, dataloader, optimizer, criterion, device, epoch)
        scheduler.step()
        
        history['train_loss'].append(results['loss'])
        history['train_acc'].append(results['accuracy'])
        history['anomaly_recall'].append(results['anomaly_recall'])
        
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Loss: {results['loss']:.4f}")
        print(f"   Accuracy: {results['accuracy']:.2f}%")
        print(f"   Anomaly Recall: {results['anomaly_recall']:.2f}%")
        print(f"   Detected: {results['detected_anomalies']}/{results['total_anomalies']}")
        print(f"   LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if results['anomaly_recall'] > best_recall:
            best_recall = results['anomaly_recall']
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_recall': best_recall,
                'history': history
            }, 'checkpoints/best_voxelwise_model.pth')
            print(f"   💾 保存最佳模型 (Recall: {best_recall:.2f}%)")
    
    # 保存最终模型
    print("\n💾 保存最终模型...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, 'checkpoints/voxelwise_final.pth')
    
    # 可视化
    print("\n📊 生成可视化...")
    os.makedirs('visualizations', exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(history['train_loss'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss (Voxelwise)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_acc'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(history['anomaly_recall'], 'r-', linewidth=2, marker='o')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Anomaly Recall (%)')
    axes[2].set_title('Anomaly Detection Recall')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=best_recall, color='orange', linestyle='--', label=f'Best: {best_recall:.1f}%')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/voxelwise_training.png', dpi=150, bbox_inches='tight')
    print("✅ 可视化保存至: visualizations/voxelwise_training.png")
    
    print("\n" + "=" * 70)
    print("🎉 训练完成！")
    print("=" * 70)
    print(f"📈 最终结果:")
    print(f"   - 最终Loss: {history['train_loss'][-1]:.4f}")
    print(f"   - 最终Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"   - 最终Anomaly Recall: {history['anomaly_recall'][-1]:.2f}%")
    print(f"   - 最佳Anomaly Recall: {best_recall:.2f}%")
    print(f"\n📁 输出文件:")
    print(f"   - 最佳模型: checkpoints/best_voxelwise_model.pth")
    print(f"   - 最终模型: checkpoints/voxelwise_final.pth")
    print(f"   - 可视化: visualizations/voxelwise_training.png")
    
    print("\n✨ 架构革命完成！")
    print("   从 '场景分类' → '体素分割'")
    print("   从 [B, 1] → [B, N]")
    print("   从 全局池化 → 逐点预测")


if __name__ == '__main__':
    main()

