"""
在测试集上评估训练好的模型
测试集使用AnoVox_Dynamic_Mono_Town07的独立子集
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from muvo.dataset.anovox_dataset import AnoVoxDataset, collate_fn
from muvo.config import _C

# 导入场景级模型
import importlib.util
spec = importlib.util.spec_from_file_location("train_scene", "train_scene_level_detection.py")
train_scene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_scene)
SceneLevelAnomalyDetector = train_scene.SceneLevelAnomalyDetector

def evaluate_model(model, test_loader, device):
    """评估模型性能"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # 数据移到GPU
            batch['image'] = batch['image'].to(device)
            batch['points'] = batch['points'].to(device)
            
            # 前向传播
            outputs = model(batch)
            logits = outputs['scene_logit']
            probs = torch.sigmoid(logits).squeeze()
            preds = (probs > 0.5).float()
            
            # 提取标签
            labels = []
            for anomaly_dict in batch['anomaly_label']:
                is_alive = anomaly_dict.get('anomaly_is_alive', 'False')
                label = 1.0 if (isinstance(is_alive, str) and is_alive.lower() == 'true') else 0.0
                labels.append(label)
            labels = torch.tensor(labels, device=device)
            
            # 收集结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, zero_division=0) * 100
    
    # 计算AUROC（如果有两个类别）
    if len(np.unique(all_labels)) > 1:
        auroc = roc_auc_score(all_labels, all_probs) * 100
    else:
        auroc = None
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    elif cm.size == 1:
        # 只有一个类别的情况
        if all_labels[0] == 1:
            tp = int(cm[0, 0])
            tn, fp, fn = 0, 0, 0
        else:
            tn = int(cm[0, 0])
            tp, fp, fn = 0, 0, 0
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'confusion_matrix': {
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn)
        },
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

def main():
    print("=" * 80)
    print("🧪 测试集评估")
    print("=" * 80)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 使用设备: {device}")
    
    # 加载测试数据集
    # 使用Dynamic_Mono_Town07的独立测试集
    print("\n📂 加载测试数据集...")
    test_dataset = AnoVoxDataset(
        data_root='/root/autodl-tmp/datasets/AnoVox',
        split='test',
        dataset_types=['Dynamic_Mono_Town07'],  # 纯异常测试集
        load_anomaly_labels=True,
        load_voxel=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn  # 使用自定义collate函数
    )
    
    print(f"✅ 测试集大小: {len(test_dataset)} 样本")
    print(f"📦 测试批次数: {len(test_loader)}")
    
    # 加载模型
    print("\n🤖 加载训练好的模型...")
    cfg = _C.clone()
    model = SceneLevelAnomalyDetector(cfg).to(device)
    
    checkpoint_path = 'checkpoints/scene_level_best.pth'
    if not Path(checkpoint_path).exists():
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ 模型加载成功: {checkpoint_path}")
    print(f"   训练Epoch: {checkpoint.get('epoch', 'N/A')}")
    best_recall = checkpoint.get('best_recall', None)
    if best_recall is not None:
        print(f"   训练Recall: {best_recall:.2f}%")
    
    # 评估
    print("\n🔬 开始测试...")
    results = evaluate_model(model, test_loader, device)
    
    # 打印结果
    print("\n" + "=" * 80)
    print("📊 测试集评估结果")
    print("=" * 80)
    print(f"\n🎯 核心指标:")
    print(f"   Accuracy:  {results['accuracy']:.2f}%")
    print(f"   Precision: {results['precision']:.2f}%")
    print(f"   Recall:    {results['recall']:.2f}%")
    print(f"   F1-Score:  {results['f1']:.2f}%")
    if results['auroc'] is not None:
        print(f"   AUROC:     {results['auroc']:.2f}%")
    
    print(f"\n🎯 混淆矩阵:")
    cm = results['confusion_matrix']
    print(f"   True Positive (TP):  {cm['TP']}")
    print(f"   True Negative (TN):  {cm['TN']}")
    print(f"   False Positive (FP): {cm['FP']}")
    print(f"   False Negative (FN): {cm['FN']}")
    
    # 计算误报率和漏检率
    fpr = cm['FP'] / (cm['FP'] + cm['TN']) * 100 if (cm['FP'] + cm['TN']) > 0 else 0
    fnr = cm['FN'] / (cm['FN'] + cm['TP']) * 100 if (cm['FN'] + cm['TP']) > 0 else 0
    
    print(f"\n⚠️ 错误分析:")
    print(f"   False Positive Rate (FPR): {fpr:.2f}%")
    print(f"   False Negative Rate (FNR): {fnr:.2f}%")
    
    # 保存结果
    output_dir = Path('test_results')
    output_dir.mkdir(exist_ok=True)
    
    np.savez(
        output_dir / 'test_results.npz',
        predictions=results['predictions'],
        labels=results['labels'],
        probabilities=results['probabilities'],
        metrics={
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'auroc': results['auroc']
        }
    )
    
    # 保存文本报告
    with open(output_dir / 'test_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("测试集评估报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"测试集大小: {len(test_dataset)} 样本\n\n")
        f.write(f"核心指标:\n")
        f.write(f"  Accuracy:  {results['accuracy']:.2f}%\n")
        f.write(f"  Precision: {results['precision']:.2f}%\n")
        f.write(f"  Recall:    {results['recall']:.2f}%\n")
        f.write(f"  F1-Score:  {results['f1']:.2f}%\n")
        if results['auroc'] is not None:
            f.write(f"  AUROC:     {results['auroc']:.2f}%\n")
        f.write(f"\n混淆矩阵:\n")
        f.write(f"  TP: {cm['TP']}, TN: {cm['TN']}, FP: {cm['FP']}, FN: {cm['FN']}\n")
        f.write(f"\n错误分析:\n")
        f.write(f"  FPR: {fpr:.2f}%\n")
        f.write(f"  FNR: {fnr:.2f}%\n")
    
    print(f"\n✅ 结果已保存到: {output_dir}/")
    print("=" * 80)

if __name__ == "__main__":
    main()

