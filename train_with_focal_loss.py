#!/usr/bin/env python3
"""
使用Focal Loss + 预训练ResNet的真正异常检测训练
解决类别不平衡问题
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


class FocalLoss(nn.Module):
    """
    Focal Loss - 专门解决类别不平衡问题
    论文: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: 平衡正负样本的权重 (0-1之间)
            gamma: 聚焦参数，越大越关注难分样本
            reduction: 'mean' 或 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [B] - 模型输出的logits (未经过sigmoid)
            targets: [B] - 真实标签 (0或1)
        """
        # 计算BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 计算pt (预测概率)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Focal Loss = -alpha * (1-pt)^gamma * log(pt)
        focal_weight = (1 - pt) ** self.gamma
        
        # 应用alpha权重
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


class ImprovedAnomalyDetector(nn.Module):
    """
    改进的异常检测模型
    - 使用预训练的ResNet18作为图像编码器
    - 保留PointNet风格的点云编码器
    - 跨模态注意力融合
    """
    def __init__(self, freeze_backbone=True):
        super().__init__()
        
        # ========== 图像编码器 (预训练ResNet18) ==========
        print("🔧 加载预训练ResNet18...")
        weights = ResNet18_Weights.IMAGENET1K_V1
        pretrained_resnet = resnet18(weights=weights)
        
        # 去掉最后的avgpool和fc层，保留特征提取部分
        # ResNet18输出: [B, 512, 7, 7] (对于224x224输入)
        self.image_encoder = nn.Sequential(*list(pretrained_resnet.children())[:-2])
        
        # 冻结backbone (可选)
        if freeze_backbone:
            print("   ❄️  冻结ResNet权重")
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        else:
            print("   🔥 解冻ResNet权重 (微调)")
        
        # 图像特征维度: 512 (ResNet18的输出通道数)
        img_feat_dim = 512
        
        # ========== 点云编码器 (PointNet风格) ==========
        # TODO: 后续可替换为Cylinder3D预训练模型
        self.point_encoder = nn.Sequential(
            nn.Linear(4, 64),  # x,y,z,intensity
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512)  # 匹配图像特征维度
        )
        
        point_feat_dim = 512
        
        # ========== 跨模态注意力融合 ==========
        fusion_dim = 512  # 统一特征维度
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,  # 增加到8个头
            batch_first=True,
            dropout=0.1
        )
        
        # ========== 异常检测头 ==========
        # 输入: 融合后的点云特征 + 图像特征
        self.anomaly_head = nn.Sequential(
            nn.Linear(fusion_dim * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, batch):
        """
        前向传播
        Args:
            batch: {'image': [B,3,H,W], 'points': [B,N,4]}
        Returns:
            {'anomaly_logits': [B,1], 'anomaly_score': [B,1]}
        """
        # 1. 图像特征提取 (预训练ResNet)
        img = batch['image']  # [B, 3, 224, 224]
        img_feat = self.image_encoder(img)  # [B, 512, 7, 7]
        
        # 展平空间维度
        B, C, H, W = img_feat.shape
        img_feat = img_feat.view(B, C, H*W)  # [B, 512, 49]
        img_feat = img_feat.permute(0, 2, 1)  # [B, 49, 512]
        
        # 2. 点云特征提取
        points = batch['points']  # [B, N, 4]
        B, N, _ = points.shape
        
        # 限制点云数量
        max_points = 2048
        if N > max_points:
            # 随机采样
            indices = torch.randperm(N, device=points.device)[:max_points]
            points = points[:, indices, :]
            N = max_points
        elif N < max_points:
            # Padding
            padding = torch.zeros(B, max_points - N, 4, device=points.device)
            points = torch.cat([points, padding], dim=1)
            N = max_points
        
        # PointNet: 对每个点独立编码
        points_reshaped = points.view(B * N, 4)  # [B*N, 4]
        point_feat = self.point_encoder(points_reshaped)  # [B*N, 512]
        point_feat = point_feat.view(B, N, 512)  # [B, N, 512]
        
        # 3. 跨模态注意力融合
        # 点云特征作为query，查询图像特征
        fused_feat, attn_weights = self.cross_attention(
            query=point_feat,     # [B, N, 512]
            key=img_feat,         # [B, 49, 512]
            value=img_feat
        )  # [B, N, 512]
        
        # 4. 全局特征聚合
        # 点云特征: max pooling
        point_global = torch.max(fused_feat, dim=1)[0]  # [B, 512]
        # 图像特征: mean pooling
        img_global = torch.mean(img_feat, dim=1)  # [B, 512]
        
        # 拼接
        combined = torch.cat([point_global, img_global], dim=1)  # [B, 1024]
        
        # 5. 异常检测
        anomaly_logits = self.anomaly_head(combined).squeeze(-1)  # [B]
        anomaly_score = torch.sigmoid(anomaly_logits)
        
        return {
            'anomaly_logits': anomaly_logits,
            'anomaly_score': anomaly_score,
            'attention_weights': attn_weights  # 用于可视化
        }


def create_pseudo_labels(batch):
    """从voxel标签创建场景级标签"""
    if 'voxel_label' in batch and batch['voxel_label'] is not None:
        labels = []
        for i in range(len(batch['voxel_label'])):
            voxel_labels = batch['voxel_label'][i]
            if isinstance(voxel_labels, torch.Tensor):
                has_anomaly = (voxel_labels > 0).any().float()
            else:
                has_anomaly = 0.0
            labels.append(has_anomaly)
        return torch.tensor(labels, dtype=torch.float32)
    else:
        # 10%随机异常
        B = batch['image'].size(0)
        return (torch.rand(B) < 0.1).float()


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    # 统计异常样本
    total_anomalies = 0
    detected_anomalies = 0
    
    all_losses = []
    all_scores = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        batch['image'] = batch['image'].to(device)
        batch['points'] = batch['points'].to(device)
        
        labels = create_pseudo_labels(batch).to(device)
        
        optimizer.zero_grad()
        outputs = model(batch)
        
        # Focal Loss
        logits = outputs['anomaly_logits']
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        # 统计
        with torch.no_grad():
            scores = outputs['anomaly_score']
            predictions = (scores > 0.5).float()
            correct = (predictions == labels).sum().item()
            
            # 统计异常检测
            anomaly_mask = (labels == 1)
            total_anomalies += anomaly_mask.sum().item()
            detected_anomalies += ((predictions == 1) & (labels == 1)).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)
            
            all_losses.append(loss.item())
            all_scores.extend(scores.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
        
        # 更新进度条
        accuracy = 100.0 * correct / labels.size(0)
        anomaly_recall = 100.0 * detected_anomalies / max(total_anomalies, 1)
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.1f}%',
            'anomaly_recall': f'{anomaly_recall:.1f}%',
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
        'all_losses': all_losses,
        'all_scores': all_scores,
        'all_labels': all_labels
    }


def main():
    """主训练函数"""
    print("=" * 70)
    print("🚀 使用Focal Loss + 预训练ResNet的异常检测训练")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # 数据集
    print("\n📁 加载数据...")
    dataset = AnoVoxDataset(
        data_root='/root/autodl-tmp/datasets/AnoVox/AnoVox_Dynamic_Mono_Town07',
        split='train'
    )
    print(f"✅ 数据加载完成: {len(dataset)} 样本")
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 模型
    print("\n🏗️ 初始化模型...")
    model = ImprovedAnomalyDetector(freeze_backbone=True).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ 总参数: {num_params:,}")
    print(f"✅ 可训练参数: {num_trainable:,}")
    print(f"✅ 冻结参数: {num_params - num_trainable:,}")
    
    # Focal Loss (关键！)
    print("\n⚡ 使用Focal Loss解决类别不平衡")
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    print(f"   alpha={0.25} (异常样本权重)")
    print(f"   gamma={2.0} (聚焦难分样本)")
    
    # 优化器
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=1e-6
    )
    
    # 训练
    num_epochs = 30
    print(f"\n🎓 开始训练 ({num_epochs} epochs)...")
    print(f"   Batch size: 8")
    print(f"   Total batches per epoch: {len(dataloader)}")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'anomaly_recall': [],
        'epoch_losses': [],
        'epoch_scores': [],
        'epoch_labels': []
    }
    
    best_recall = 0.0
    
    for epoch in range(1, num_epochs + 1):
        results = train_epoch(model, dataloader, optimizer, criterion, device, epoch)
        scheduler.step()
        
        history['train_loss'].append(results['loss'])
        history['train_acc'].append(results['accuracy'])
        history['anomaly_recall'].append(results['anomaly_recall'])
        history['epoch_losses'].append(results['all_losses'])
        history['epoch_scores'].append(results['all_scores'])
        history['epoch_labels'].append(results['all_labels'])
        
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
            }, 'checkpoints/best_focal_loss_model.pth')
            print(f"   💾 保存最佳模型 (Recall: {best_recall:.2f}%)")
    
    # 保存最终模型
    print("\n💾 保存最终模型...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, 'checkpoints/focal_loss_final.pth')
    
    # 可视化
    print("\n📊 生成可视化...")
    os.makedirs('visualizations', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss曲线
    axes[0, 0].plot(history['train_loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss (Focal Loss)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy曲线
    axes[0, 1].plot(history['train_acc'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Anomaly Recall曲线 (关键指标!)
    axes[1, 0].plot(history['anomaly_recall'], 'r-', linewidth=2, marker='o')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Anomaly Recall (%)')
    axes[1, 0].set_title('Anomaly Detection Recall (关键指标)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=best_recall, color='orange', linestyle='--', label=f'Best: {best_recall:.1f}%')
    axes[1, 0].legend()
    
    # 分数分布
    last_scores = np.array(history['epoch_scores'][-1])
    last_labels = np.array(history['epoch_labels'][-1])
    
    normal_scores = last_scores[last_labels == 0]
    anomaly_scores = last_scores[last_labels == 1]
    
    if len(normal_scores) > 0:
        axes[1, 1].hist(normal_scores, bins=50, alpha=0.6, color='green', label='Normal')
    if len(anomaly_scores) > 0:
        axes[1, 1].hist(anomaly_scores, bins=50, alpha=0.6, color='red', label='Anomaly')
    
    axes[1, 1].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    axes[1, 1].set_xlabel('Anomaly Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Score Distribution (Epoch {num_epochs})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/focal_loss_training.png', dpi=150, bbox_inches='tight')
    print("✅ 可视化保存至: visualizations/focal_loss_training.png")
    
    print("\n" + "=" * 70)
    print("🎉 训练完成！")
    print("=" * 70)
    print(f"📈 最终结果:")
    print(f"   - 最终Loss: {history['train_loss'][-1]:.4f}")
    print(f"   - 最终Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"   - 最终Anomaly Recall: {history['anomaly_recall'][-1]:.2f}%")
    print(f"   - 最佳Anomaly Recall: {best_recall:.2f}%")
    print(f"\n📁 输出文件:")
    print(f"   - 最佳模型: checkpoints/best_focal_loss_model.pth")
    print(f"   - 最终模型: checkpoints/focal_loss_final.pth")
    print(f"   - 可视化: visualizations/focal_loss_training.png")


if __name__ == '__main__':
    main()

