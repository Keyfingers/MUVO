#!/usr/bin/env python3
"""
评估Focal Loss训练的模型
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt

sys.path.insert(0, '/root/autodl-tmp/MUVO/MUVO')

from muvo.dataset.anovox_dataset import AnoVoxDataset, collate_fn
from train_with_focal_loss import ImprovedAnomalyDetector, create_pseudo_labels


def evaluate_model(model, dataloader, device):
    """评估模型"""
    model.eval()
    
    all_scores = []
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            batch['image'] = batch['image'].to(device)
            batch['points'] = batch['points'].to(device)
            
            labels = create_pseudo_labels(batch).to(device)
            outputs = model(batch)
            
            scores = outputs['anomaly_score'].cpu().numpy()
            labels_np = labels.cpu().numpy()
            predictions = (scores > 0.5).astype(float)
            
            all_scores.extend(scores.tolist())
            all_labels.extend(labels_np.tolist())
            all_predictions.extend(predictions.tolist())
    
    return np.array(all_scores), np.array(all_labels), np.array(all_predictions)


def compute_metrics(scores, labels, predictions):
    """计算指标"""
    # 基础指标
    accuracy = (predictions == labels).mean()
    
    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # AUROC和AUPRC
    try:
        auroc = roc_auc_score(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)
    except:
        auroc = 0.0
        fpr, tpr = None, None
    
    try:
        auprc = average_precision_score(labels, scores)
        precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
    except:
        auprc = 0.0
        precision_curve, recall_curve = None, None
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc,
        'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)},
        'roc_curve': (fpr, tpr),
        'pr_curve': (precision_curve, recall_curve)
    }


def visualize_comparison(metrics_before, metrics_after, scores_after, labels_after):
    """对比可视化"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 指标对比
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUROC', 'AUPRC']
    before_values = [
        metrics_before['accuracy'],
        metrics_before['precision'],
        metrics_before['recall'],
        metrics_before['f1'],
        metrics_before['auroc'],
        metrics_before['auprc']
    ]
    after_values = [
        metrics_after['accuracy'],
        metrics_after['precision'],
        metrics_after['recall'],
        metrics_after['f1'],
        metrics_after['auroc'],
        metrics_after['auprc']
    ]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, before_values, width, label='Before (BCE)', alpha=0.7, color='red')
    axes[0, 0].bar(x + width/2, after_values, width, label='After (Focal)', alpha=0.7, color='green')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Metrics Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metric_names, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 1])
    
    # 2. ROC曲线对比
    if metrics_before['roc_curve'][0] is not None:
        fpr_b, tpr_b = metrics_before['roc_curve']
        axes[0, 1].plot(fpr_b, tpr_b, 'r-', linewidth=2, label=f'Before (AUROC={metrics_before["auroc"]:.3f})')
    if metrics_after['roc_curve'][0] is not None:
        fpr_a, tpr_a = metrics_after['roc_curve']
        axes[0, 1].plot(fpr_a, tpr_a, 'g-', linewidth=2, label=f'After (AUROC={metrics_after["auroc"]:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 混淆矩阵对比
    cm_before = metrics_before['confusion_matrix']
    cm_after = metrics_after['confusion_matrix']
    
    cm_text = f"""
    Before (BCE Loss):
      TP: {cm_before['tp']:6d}  FP: {cm_before['fp']:6d}
      FN: {cm_before['fn']:6d}  TN: {cm_before['tn']:6d}
      Recall: {metrics_before['recall']:.2%}
    
    After (Focal Loss):
      TP: {cm_after['tp']:6d}  FP: {cm_after['fp']:6d}
      FN: {cm_after['fn']:6d}  TN: {cm_after['tn']:6d}
      Recall: {metrics_after['recall']:.2%}
    """
    axes[0, 2].text(0.1, 0.5, cm_text, fontsize=11, family='monospace', verticalalignment='center')
    axes[0, 2].axis('off')
    axes[0, 2].set_title('Confusion Matrix Comparison')
    
    # 4. 分数分布
    normal_scores = scores_after[labels_after == 0]
    anomaly_scores = scores_after[labels_after == 1]
    
    if len(normal_scores) > 0:
        axes[1, 0].hist(normal_scores, bins=50, alpha=0.6, color='green', label='Normal')
    if len(anomaly_scores) > 0:
        axes[1, 0].hist(anomaly_scores, bins=50, alpha=0.6, color='red', label='Anomaly')
    axes[1, 0].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    axes[1, 0].set_xlabel('Anomaly Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Score Distribution (Focal Loss)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 分数统计
    stats_text = f"""
    Score Statistics:
    
    Normal samples:
      Mean: {normal_scores.mean():.4f}
      Std:  {normal_scores.std():.4f}
      Min:  {normal_scores.min():.4f}
      Max:  {normal_scores.max():.4f}
    
    Anomaly samples:
      Mean: {anomaly_scores.mean():.4f}
      Std:  {anomaly_scores.std():.4f}
      Min:  {anomaly_scores.min():.4f}
      Max:  {anomaly_scores.max():.4f}
    
    Separation:
      Delta Mean: {abs(anomaly_scores.mean() - normal_scores.mean()):.4f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace', verticalalignment='center')
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Score Statistics')
    
    # 6. 关键问题诊断
    diagnosis_text = f"""
    Diagnosis:
    
    Problem: {"SAME AS BEFORE! 模型仍在投机取巧!" if metrics_after['auroc'] < 0.55 else "Improved!"}
    
    Evidence:
      1. TP = {cm_after['tp']} {"❌ (Still 0!)" if cm_after['tp'] == 0 else "✅"}
      2. Recall = {metrics_after['recall']:.1%} {"❌" if metrics_after['recall'] < 0.05 else "✅"}
      3. AUROC = {metrics_after['auroc']:.3f} {"❌ (Random!)" if metrics_after['auroc'] < 0.55 else "✅"}
    
    Root Cause:
      {"1. Focal Loss参数可能不够激进" if cm_after['tp'] == 0 else ""}
      {"2. 伪标签质量太差" if cm_after['tp'] == 0 else ""}
      {"3. 需要更强的监督信号" if cm_after['tp'] == 0 else ""}
    
    Next Steps:
      {"- 增大alpha (0.25→0.1)" if cm_after['tp'] == 0 else ""}
      {"- 增大gamma (2.0→3.0)" if cm_after['tp'] == 0 else ""}
      {"- 使用体素级标签" if cm_after['tp'] == 0 else ""}
      {"- 加大异常样本权重" if cm_after['tp'] == 0 else ""}
    """
    axes[1, 2].text(0.05, 0.5, diagnosis_text, fontsize=9, family='monospace', verticalalignment='center')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Diagnosis & Next Steps', color='red' if metrics_after['auroc'] < 0.55 else 'green')
    
    plt.tight_layout()
    plt.savefig('visualizations/focal_loss_evaluation.png', dpi=150, bbox_inches='tight')
    print("✅ 对比可视化保存至: visualizations/focal_loss_evaluation.png")


def main():
    print("=" * 70)
    print("📊 Focal Loss模型评估与对比")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 使用设备: {device}")
    
    # 加载数据
    print("\n📁 加载数据...")
    dataset = AnoVoxDataset(
        data_root='/root/autodl-tmp/datasets/AnoVox/AnoVox_Dynamic_Mono_Town07',
        split='train'
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # 加载Focal Loss模型
    print("\n🏗️ 加载Focal Loss模型...")
    model = ImprovedAnomalyDetector(freeze_backbone=True).to(device)
    
    if os.path.exists('checkpoints/focal_loss_final.pth'):
        checkpoint = torch.load('checkpoints/focal_loss_final.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ 加载最终模型")
    else:
        print("❌ 模型文件不存在")
        return
    
    # 评估
    print("\n🔍 评估Focal Loss模型...")
    scores_after, labels_after, predictions_after = evaluate_model(model, dataloader, device)
    metrics_after = compute_metrics(scores_after, labels_after, predictions_after)
    
    print("\n✅ Focal Loss模型结果:")
    print(f"   Accuracy:  {metrics_after['accuracy']:.4f}")
    print(f"   Precision: {metrics_after['precision']:.4f}")
    print(f"   Recall:    {metrics_after['recall']:.4f}")
    print(f"   F1:        {metrics_after['f1']:.4f}")
    print(f"   AUROC:     {metrics_after['auroc']:.4f}")
    print(f"   AUPRC:     {metrics_after['auprc']:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"   TP: {metrics_after['confusion_matrix']['tp']:6d}  FP: {metrics_after['confusion_matrix']['fp']:6d}")
    print(f"   FN: {metrics_after['confusion_matrix']['fn']:6d}  TN: {metrics_after['confusion_matrix']['tn']:6d}")
    
    # 之前的BCE模型结果(从日志中)
    print("\n📊 之前的BCE Loss模型结果:")
    metrics_before = {
        'accuracy': 0.9012,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'auroc': 0.4979,
        'auprc': 0.0983,
        'confusion_matrix': {'tp': 0, 'fp': 0, 'tn': 3780, 'fn': 420},
        'roc_curve': (None, None),
        'pr_curve': (None, None)
    }
    print(f"   Accuracy:  {metrics_before['accuracy']:.4f}")
    print(f"   Precision: {metrics_before['precision']:.4f}")
    print(f"   Recall:    {metrics_before['recall']:.4f}")
    print(f"   F1:        {metrics_before['f1']:.4f}")
    print(f"   AUROC:     {metrics_before['auroc']:.4f}")
    print(f"   AUPRC:     {metrics_before['auprc']:.4f}")
    
    # 对比分析
    print("\n" + "=" * 70)
    print("📈 对比分析")
    print("=" * 70)
    
    auroc_change = metrics_after['auroc'] - metrics_before['auroc']
    recall_change = metrics_after['recall'] - metrics_before['recall']
    
    print(f"\n关键指标变化:")
    print(f"  AUROC:  {metrics_before['auroc']:.4f} → {metrics_after['auroc']:.4f} ({auroc_change:+.4f})")
    print(f"  Recall: {metrics_before['recall']:.4f} → {metrics_after['recall']:.4f} ({recall_change:+.4f})")
    print(f"  TP:     {metrics_before['confusion_matrix']['tp']} → {metrics_after['confusion_matrix']['tp']}")
    
    if metrics_after['auroc'] < 0.55 and metrics_after['confusion_matrix']['tp'] == 0:
        print("\n❌ 结论: Focal Loss未能解决问题！")
        print("   模型仍在'投机取巧' - 永远预测正常")
        print("\n根本原因:")
        print("   1. 伪标签质量太差（场景级 vs 体素级）")
        print("   2. Focal Loss参数可能不够激进")
        print("   3. 需要更强的监督信号")
    elif metrics_after['auroc'] > 0.60:
        print("\n✅ 结论: Focal Loss有效！")
        print("   模型开始真正学习异常特征")
    else:
        print("\n⚠️ 结论: Focal Loss有轻微改善")
        print("   但效果不够显著，需要进一步优化")
    
    # 可视化
    print("\n📊 生成对比可视化...")
    visualize_comparison(metrics_before, metrics_after, scores_after, labels_after)
    
    print("\n" + "=" * 70)
    print("🎉 评估完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()

