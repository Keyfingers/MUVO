"""
🔬 消融实验训练脚本

支持5种消融配置：
1. image_only    - 仅图像分支
2. lidar_only    - 仅点云分支  
3. late_fusion   - 后期融合
4. dynamic_only  - 仅异常数据训练
5. lightweight   - 轻量级模型

用法:
    python ablation_study.py --experiment image_only --epochs 20
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from muvo.config import _C
from muvo.dataset.anovox_dataset import AnoVoxDataset, collate_fn


# ============================================================================
# 模型定义
# ============================================================================

class ImageOnlyDetector(nn.Module):
    """消融实验1: 仅图像分支"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 图像编码器（与完整模型相同）
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 2),
            self._make_layer(256, 512, 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 分类器（仅512维输入）
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, batch):
        image = batch['image']
        B = image.shape[0]
        
        img_feat = self.image_encoder(image).view(B, -1)
        logit = self.classifier(img_feat)
        prob = torch.sigmoid(logit)
        
        return {'scene_logit': logit, 'scene_prob': prob}


class LiDAROnlyDetector(nn.Module):
    """消融实验2: 仅点云分支"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 点云编码器（与完整模型相同）
        self.point_encoder = nn.Sequential(
            nn.Conv1d(4, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        
        # 分类器（仅512维输入）
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, batch):
        points = batch['points']
        B = points.shape[0]
        
        # 限制点云数量以避免内存溢出（点云可能非常大）
        max_points = 16384  # 限制最多16384个点
        N = points.shape[1]
        
        if N > max_points:
            # 随机采样
            indices = torch.randperm(N, device=points.device)[:max_points]
            points = points[:, indices, :]
        elif N < max_points:
            # 如果点数不足，可以重复采样或填充（这里简单处理）
            pass
        
        points_t = points.permute(0, 2, 1)  # [B, 4, N]
        point_feat = torch.max(self.point_encoder(points_t), 2)[0]  # [B, 512]
        logit = self.classifier(point_feat)
        prob = torch.sigmoid(logit)
        
        return {'scene_logit': logit, 'scene_prob': prob}


class LateFusionDetector(nn.Module):
    """消融实验3: 后期融合（两个独立分类器的平均）"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 图像分支
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 2),
            self._make_layer(256, 512, 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 点云分支
        self.point_encoder = nn.Sequential(
            nn.Conv1d(4, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        
        # 两个独立的分类器
        self.image_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
        self.point_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, batch):
        image = batch['image']
        points = batch['points']
        B = image.shape[0]
        
        # 图像分支预测
        img_feat = self.image_encoder(image).view(B, -1)
        img_logit = self.image_classifier(img_feat)
        
        # 点云分支预测
        points_t = points.permute(0, 2, 1)
        point_feat = torch.max(self.point_encoder(points_t), 2)[0]
        point_logit = self.point_classifier(point_feat)
        
        # 后期融合：平均两个预测
        fused_logit = (img_logit + point_logit) / 2.0
        fused_prob = torch.sigmoid(fused_logit)
        
        return {'scene_logit': fused_logit, 'scene_prob': fused_prob}


class LightweightDetector(nn.Module):
    """消融实验5: 轻量级模型（通道数减半）"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 轻量级图像编码器
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            self._make_layer(32, 64, 2),
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 轻量级点云编码器
        self.point_encoder = nn.Sequential(
            nn.Conv1d(4, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        
        # 轻量级分类器
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, batch):
        image = batch['image']
        points = batch['points']
        B = image.shape[0]
        
        img_feat = self.image_encoder(image).view(B, -1)
        points_t = points.permute(0, 2, 1)
        point_feat = torch.max(self.point_encoder(points_t), 2)[0]
        
        fused_feat = torch.cat([img_feat, point_feat], dim=1)
        logit = self.classifier(fused_feat)
        prob = torch.sigmoid(logit)
        
        return {'scene_logit': logit, 'scene_prob': prob}


# ============================================================================
# 辅助函数
# ============================================================================

def extract_scene_labels(batch):
    """从batch中提取场景级标签"""
    anomaly_labels = batch.get('anomaly_label', [])
    labels = []
    for label_dict in anomaly_labels:
        if isinstance(label_dict, dict):
            anomaly_is_alive = label_dict.get('anomaly_is_alive', 'False')
            if isinstance(anomaly_is_alive, str):
                has_anomaly = (anomaly_is_alive.lower() == 'true')
            else:
                has_anomaly = bool(anomaly_is_alive)
            labels.append(1.0 if has_anomaly else 0.0)
        else:
            labels.append(0.0)
    return torch.tensor(labels, dtype=torch.float32)


def calculate_metrics(preds, labels):
    """计算分类指标"""
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# 训练函数
# ============================================================================

def train_ablation(experiment_name, epochs=20, batch_size=8, lr=1e-4):
    """
    运行消融实验
    
    Args:
        experiment_name: 实验名称 (image_only, lidar_only, late_fusion, dynamic_only, lightweight)
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
    """
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建输出目录
    output_dir = Path(f'ablation_results/{experiment_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = _C.clone()
    
    print(f"\n{'='*80}")
    print(f"🔬 消融实验: {experiment_name}")
    print(f"{'='*80}\n")
    
    # 准备数据集
    if experiment_name == 'dynamic_only':
        # 仅使用异常数据
        dataset_types_train = ['Dynamic_Mono_Town07']
        dataset_types_val = ['Dynamic_Mono_Town07']
    else:
        # 使用混合数据
        dataset_types_train = ['Dynamic_Mono_Town07', 'Normality_Mono_Town07']
        dataset_types_val = ['Dynamic_Mono_Town07', 'Normality_Mono_Town07']
    
    train_dataset = AnoVoxDataset(
        data_root='/root/autodl-tmp/datasets/AnoVox',
        split='train',
        dataset_types=dataset_types_train,
        train_ratio=0.8,
        load_voxel=False
    )
    
    val_dataset = AnoVoxDataset(
        data_root='/root/autodl-tmp/datasets/AnoVox',
        split='val',
        dataset_types=dataset_types_val,
        train_ratio=0.8,
        load_voxel=False
    )
    
    # 为lidar_only使用更小的batch_size以避免内存溢出
    effective_batch_size = batch_size
    if experiment_name == 'lidar_only':
        effective_batch_size = min(batch_size, 4)  # lidar_only最多使用batch_size=4
        print(f"⚠️  lidar_only实验：使用batch_size={effective_batch_size}以避免内存溢出")
    
    train_loader = DataLoader(
        train_dataset, batch_size=effective_batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=effective_batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn
    )
    
    # 创建模型
    if experiment_name == 'image_only':
        model = ImageOnlyDetector(cfg)
    elif experiment_name == 'lidar_only':
        model = LiDAROnlyDetector(cfg)
    elif experiment_name == 'late_fusion':
        model = LateFusionDetector(cfg)
    elif experiment_name == 'dynamic_only':
        # dynamic_only使用完整模型，但仅用异常数据训练
        from train_scene_level_detection import SceneLevelAnomalyDetector
        model = SceneLevelAnomalyDetector(cfg)
    elif experiment_name == 'lightweight':
        model = LightweightDetector(cfg)
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    model = model.to(device)
    
    # 统计参数量
    n_params = count_parameters(model)
    print(f"📊 模型参数量: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # 统计数据分布
    print(f"📂 训练集: {len(train_dataset)} 样本")
    print(f"📂 验证集: {len(val_dataset)} 样本")
    
    # 统计标签分布（随机采样1000个样本，确保混合）
    import random
    random.seed(42)
    sample_indices = random.sample(range(len(train_dataset)), min(1000, len(train_dataset)))
    anomaly_count = 0
    normal_count = 0
    for i in sample_indices:
        sample = train_dataset[i]
        label_dict = sample.get('anomaly_label', {})
        if isinstance(label_dict, dict):
            is_alive = label_dict.get('anomaly_is_alive', 'False')
            if isinstance(is_alive, str):
                has_anomaly = (is_alive.lower() == 'true')
            else:
                has_anomaly = bool(is_alive)
            if has_anomaly:
                anomaly_count += 1
            else:
                normal_count += 1
    print(f"📊 训练集标签分布 (随机1000样本): 异常={anomaly_count}, 正常={normal_count} ({anomaly_count/(anomaly_count+normal_count)*100:.1f}% 异常)\n")
    
    # 定义损失函数和优化器
    pos_weight = torch.tensor([3.5]).to(device)  # 77.8/22.2 ≈ 3.5
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 训练记录
    history = {
        'train_loss': [], 'val_loss': [],
        'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1': [],
        'val_tp': [], 'val_tn': [], 'val_fp': [], 'val_fn': []
    }
    
    best_recall = 0.0
    
    # 训练循环
    for epoch in range(epochs):
        # ========== 训练 ==========
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch in pbar:
            # 准备数据
            for key in ['image', 'points']:
                if key in batch:
                    batch[key] = batch[key].to(device)
            
            labels = extract_scene_labels(batch).to(device)
            
            # 前向传播
            output = model(batch)
            logits = output['scene_logit'].squeeze()
            
            # 计算损失
            loss = criterion(logits, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # ========== 验证 ==========
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                for key in ['image', 'points']:
                    if key in batch:
                        batch[key] = batch[key].to(device)
                
                labels = extract_scene_labels(batch).to(device)
                
                output = model(batch)
                logits = output['scene_logit'].squeeze()
                probs = output['scene_prob'].squeeze()
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = (probs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # 计算指标
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        metrics = calculate_metrics(all_preds, all_labels)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        for key in ['accuracy', 'precision', 'recall', 'f1', 'tp', 'tn', 'fp', 'fn']:
            history[f'val_{key}'].append(metrics[key])
        
        # 打印结果
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Accuracy:   {metrics['accuracy']:.4f}")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print(f"  F1-Score:   {metrics['f1']:.4f}")
        print(f"  TP: {metrics['tp']}, TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}\n")
        
        # 保存最佳模型
        if metrics['recall'] > best_recall:
            best_recall = metrics['recall']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics
            }, output_dir / 'best_model.pth')
            print(f"  ✅ 保存最佳模型 (Recall: {best_recall:.4f})")
    
    # 保存训练历史
    np.savez(output_dir / 'history.npz', **history)
    
    # 保存最终报告
    final_metrics = {
        'experiment': experiment_name,
        'epochs': epochs,
        'best_recall': best_recall,
        'final_metrics': {k: float(v) for k, v in metrics.items()},
        'n_parameters': n_params,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset)
    }
    
    with open(output_dir / 'report.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ 实验完成: {experiment_name}")
    print(f"   最佳 Recall: {best_recall:.4f}")
    print(f"   结果保存至: {output_dir}")
    print(f"{'='*80}\n")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='消融实验训练脚本')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['image_only', 'lidar_only', 'late_fusion', 
                                'dynamic_only', 'lightweight'],
                        help='实验类型')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    
    args = parser.parse_args()
    
    train_ablation(
        experiment_name=args.experiment,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )


if __name__ == '__main__':
    main()
