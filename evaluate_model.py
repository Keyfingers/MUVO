#!/usr/bin/env python3
"""
模型评估脚本 - 计算AUROC和AUPRC指标
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

sys.path.insert(0, '/root/autodl-tmp/MUVO/MUVO')

from muvo.dataset.anovox_dataset import AnoVoxDataset, collate_fn
from train_anomaly_with_labels import AnomalyDetectionModel, create_pseudo_labels


def evaluate_model(model, dataloader, device):
    """评估模型性能"""
    model.eval()
    
    all_scores = []
    all_labels = []
    all_predictions = []
    
    print("\n🔍 评估模型...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            # 移动数据到设备
            batch['image'] = batch['image'].to(device)
            batch['points'] = batch['points'].to(device)
            
            # 创建标签
            labels = create_pseudo_labels(batch).to(device)
            
            # 前向传播
            outputs = model(batch)
            scores = outputs['anomaly_score'].squeeze(-1)
            predictions = (scores > 0.5).float()
            
            # 收集结果
            all_scores.extend(scores.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
    
    # 转换为numpy数组
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    return all_scores, all_labels, all_predictions


def compute_metrics(scores, labels, predictions):
    """计算评估指标"""
    print("\n📊 计算评估指标...")
    
    # 基础指标
    accuracy = (predictions == labels).mean()
    
    # 混淆矩阵
    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    tn = ((predictions == 0) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # AUROC和AUPRC
    try:
        auroc = roc_auc_score(labels, scores)
    except:
        auroc = 0.0
        print("⚠️ 无法计算AUROC（可能标签分布不均）")
    
    try:
        auprc = average_precision_score(labels, scores)
    except:
        auprc = 0.0
        print("⚠️ 无法计算AUPRC（可能标签分布不均）")
    
    # ROC曲线
    try:
        fpr, tpr, thresholds_roc = roc_curve(labels, scores)
    except:
        fpr, tpr, thresholds_roc = None, None, None
    
    # PR曲线
    try:
        precision_curve, recall_curve, thresholds_pr = precision_recall_curve(labels, scores)
    except:
        precision_curve, recall_curve, thresholds_pr = None, None, None
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
        'roc_curve': (fpr, tpr, thresholds_roc),
        'pr_curve': (precision_curve, recall_curve, thresholds_pr)
    }
    
    return metrics


def visualize_results(metrics, scores, labels, split_name='test'):
    """可视化评估结果"""
    print("\n📊 生成可视化...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. ROC曲线
    if metrics['roc_curve'][0] is not None:
        fpr, tpr, _ = metrics['roc_curve']
        axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUROC = {metrics["auroc"]:.4f}')
        axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'ROC Curve Unavailable', ha='center', va='center')
        axes[0, 0].set_title('ROC Curve')
    
    # 2. PR曲线
    if metrics['pr_curve'][0] is not None:
        precision_curve, recall_curve, _ = metrics['pr_curve']
        axes[0, 1].plot(recall_curve, precision_curve, 'g-', linewidth=2, 
                       label=f'AUPRC = {metrics["auprc"]:.4f}')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'PR Curve Unavailable', ha='center', va='center')
        axes[0, 1].set_title('Precision-Recall Curve')
    
    # 3. 混淆矩阵
    cm = metrics['confusion_matrix']
    cm_array = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
    im = axes[0, 2].imshow(cm_array, cmap='Blues')
    axes[0, 2].set_xticks([0, 1])
    axes[0, 2].set_yticks([0, 1])
    axes[0, 2].set_xticklabels(['Normal', 'Anomaly'])
    axes[0, 2].set_yticklabels(['Normal', 'Anomaly'])
    axes[0, 2].set_xlabel('Predicted')
    axes[0, 2].set_ylabel('True')
    axes[0, 2].set_title('Confusion Matrix')
    
    # 添加数值标注
    for i in range(2):
        for j in range(2):
            text = axes[0, 2].text(j, i, int(cm_array[i, j]),
                                  ha="center", va="center", color="black", fontsize=14)
    plt.colorbar(im, ax=axes[0, 2])
    
    # 4. 异常分数分布
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    if len(normal_scores) > 0:
        axes[1, 0].hist(normal_scores, bins=50, alpha=0.6, color='green', label='Normal')
    if len(anomaly_scores) > 0:
        axes[1, 0].hist(anomaly_scores, bins=50, alpha=0.6, color='red', label='Anomaly')
    axes[1, 0].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    axes[1, 0].set_xlabel('Anomaly Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Score Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 指标柱状图
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUROC', 'AUPRC']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
        metrics['auroc'],
        metrics['auprc']
    ]
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
    axes[1, 1].bar(metric_names, metric_values, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标注
    for i, v in enumerate(metric_values):
        axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    # 6. 指标摘要
    axes[1, 2].axis('off')
    summary_text = f"""
    📊 Evaluation Summary ({split_name})
    
    ═══════════════════════════════
    
    Classification Metrics:
      • Accuracy:  {metrics['accuracy']:.4f}
      • Precision: {metrics['precision']:.4f}
      • Recall:    {metrics['recall']:.4f}
      • F1 Score:  {metrics['f1']:.4f}
    
    Ranking Metrics:
      • AUROC:     {metrics['auroc']:.4f}
      • AUPRC:     {metrics['auprc']:.4f}
    
    Confusion Matrix:
      • TP: {cm['tp']:>6d}  FP: {cm['fp']:>6d}
      • FN: {cm['fn']:>6d}  TN: {cm['tn']:>6d}
    
    Sample Distribution:
      • Normal:    {len(normal_scores):>6d}
      • Anomaly:   {len(anomaly_scores):>6d}
      • Total:     {len(scores):>6d}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    
    # 保存
    os.makedirs('visualizations', exist_ok=True)
    output_path = f'visualizations/evaluation_{split_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化保存至: {output_path}")
    
    return output_path


def main():
    """主评估函数"""
    print("=" * 70)
    print("🎯 MUVO异常检测模型评估")
    print("=" * 70)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 使用设备: {device}")
    
    # 加载模型
    print("\n🏗️ 加载模型...")
    model_path = 'checkpoints/real_anomaly_detector.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("   请先完成训练！")
        return
    
    model = AnomalyDetectionModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ 模型加载成功")
    
    # 评估训练集
    print("\n" + "=" * 70)
    print("📊 评估训练集")
    print("=" * 70)
    
    train_dataset = AnoVoxDataset(
        data_root='/root/autodl-tmp/datasets/AnoVox/AnoVox_Dynamic_Mono_Town07',
        split='train'
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    train_scores, train_labels, train_predictions = evaluate_model(model, train_loader, device)
    train_metrics = compute_metrics(train_scores, train_labels, train_predictions)
    
    print("\n✅ 训练集结果:")
    print(f"   Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"   AUROC:     {train_metrics['auroc']:.4f}")
    print(f"   AUPRC:     {train_metrics['auprc']:.4f}")
    
    visualize_results(train_metrics, train_scores, train_labels, 'train')
    
    # 评估验证集（如果存在）
    print("\n" + "=" * 70)
    print("📊 评估验证集")
    print("=" * 70)
    
    try:
        val_dataset = AnoVoxDataset(
            data_root='/root/autodl-tmp/datasets/AnoVox/AnoVox_Dynamic_Mono_Town07',
            split='val'
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )
        
        val_scores, val_labels, val_predictions = evaluate_model(model, val_loader, device)
        val_metrics = compute_metrics(val_scores, val_labels, val_predictions)
        
        print("\n✅ 验证集结果:")
        print(f"   Accuracy:  {val_metrics['accuracy']:.4f}")
        print(f"   AUROC:     {val_metrics['auroc']:.4f}")
        print(f"   AUPRC:     {val_metrics['auprc']:.4f}")
        
        visualize_results(val_metrics, val_scores, val_labels, 'val')
    except Exception as e:
        print(f"⚠️ 跳过验证集评估: {e}")
    
    print("\n" + "=" * 70)
    print("🎉 评估完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()

