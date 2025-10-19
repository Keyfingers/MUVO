"""
训练过程可视化脚本
生成Loss曲线、Accuracy/Recall曲线、混淆矩阵等图表
"""

import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

def parse_training_log(log_file):
    """解析训练日志"""
    epochs = []
    losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    # 混淆矩阵数据
    tps = []
    fps = []
    fns = []
    tns = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取每个epoch的Summary信息
    pattern = r'Epoch (\d+).*?Summary:.*?Loss: ([\d.]+).*?Accuracy: ([\d.]+)%.*?Precision: ([\d.]+)%.*?Recall: ([\d.]+)%.*?F1-Score: ([\d.]+)%.*?TP: (\d+), FP: (\d+), FN: (\d+), TN: (\d+)'
    
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        epoch = int(match.group(1))
        loss = float(match.group(2))
        acc = float(match.group(3))
        prec = float(match.group(4))
        rec = float(match.group(5))
        f1 = float(match.group(6))
        tp = int(match.group(7))
        fp = int(match.group(8))
        fn = int(match.group(9))
        tn = int(match.group(10))
        
        epochs.append(epoch)
        losses.append(loss)
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        tns.append(tn)
    
    return {
        'epochs': epochs,
        'losses': losses,
        'accuracies': accuracies,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
        'tps': tps,
        'fps': fps,
        'fns': fns,
        'tns': tns
    }

def plot_training_curves(data, output_dir):
    """绘制训练曲线"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. Loss曲线
    plt.figure(figsize=(10, 6))
    plt.plot(data['epochs'], data['losses'], 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title('Training Loss Curve', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Loss曲线已保存: {output_dir / 'loss_curve.png'}")
    
    # 2. Accuracy/Recall/Precision曲线
    plt.figure(figsize=(12, 6))
    plt.plot(data['epochs'], data['accuracies'], 'b-', linewidth=2, marker='o', markersize=4, label='Accuracy')
    plt.plot(data['epochs'], data['recalls'], 'r-', linewidth=2, marker='s', markersize=4, label='Recall')
    plt.plot(data['epochs'], data['precisions'], 'g-', linewidth=2, marker='^', markersize=4, label='Precision')
    plt.plot(data['epochs'], data['f1_scores'], 'm-', linewidth=2, marker='d', markersize=4, label='F1-Score')
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    plt.title('Training Metrics Over Epochs', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim([70, 101])
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 指标曲线已保存: {output_dir / 'metrics_curve.png'}")
    
    # 3. 最终混淆矩阵 (使用最后一个epoch的数据)
    plt.figure(figsize=(8, 6))
    tp, fp, fn, tn = data['tps'][-1], data['fps'][-1], data['fns'][-1], data['tns'][-1]
    cm = np.array([[tn, fp], [fn, tp]])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Predicted Normal', 'Predicted Anomaly'],
                yticklabels=['Actual Normal', 'Actual Anomaly'],
                annot_kws={'size': 16, 'weight': 'bold'})
    plt.title('Final Confusion Matrix (Epoch 30)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存: {output_dir / 'confusion_matrix.png'}")
    
    # 4. TP/FP/FN/TN随时间变化
    plt.figure(figsize=(12, 6))
    plt.plot(data['epochs'], data['tps'], 'g-', linewidth=2, marker='o', label='True Positive (TP)')
    plt.plot(data['epochs'], data['tns'], 'b-', linewidth=2, marker='s', label='True Negative (TN)')
    plt.plot(data['epochs'], data['fps'], 'r-', linewidth=2, marker='^', label='False Positive (FP)')
    plt.plot(data['epochs'], data['fns'], 'm-', linewidth=2, marker='d', label='False Negative (FN)')
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix Components Over Epochs', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_components.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵组件图已保存: {output_dir / 'confusion_components.png'}")
    
    # 5. 综合对比图 (2x2布局)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 左上: Loss
    axes[0, 0].plot(data['epochs'], data['losses'], 'b-', linewidth=2, marker='o')
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 右上: Accuracy & Recall
    axes[0, 1].plot(data['epochs'], data['accuracies'], 'b-', linewidth=2, marker='o', label='Accuracy')
    axes[0, 1].plot(data['epochs'], data['recalls'], 'r-', linewidth=2, marker='s', label='Recall')
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Accuracy & Recall', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([70, 101])
    
    # 左下: Precision & F1
    axes[1, 0].plot(data['epochs'], data['precisions'], 'g-', linewidth=2, marker='^', label='Precision')
    axes[1, 0].plot(data['epochs'], data['f1_scores'], 'm-', linewidth=2, marker='d', label='F1-Score')
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Precision & F1-Score', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([70, 101])
    
    # 右下: 混淆矩阵
    cm = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=axes[1, 1],
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'],
                annot_kws={'size': 14, 'weight': 'bold'})
    axes[1, 1].set_title('Final Confusion Matrix', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('True', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 综合汇总图已保存: {output_dir / 'training_summary.png'}")

def generate_statistics_report(data, output_dir):
    """生成统计报告"""
    output_dir = Path(output_dir)
    
    report = []
    report.append("=" * 80)
    report.append("🎯 训练统计报告")
    report.append("=" * 80)
    report.append("")
    
    # 基本信息
    report.append(f"📊 训练轮数: {len(data['epochs'])} epochs")
    report.append(f"📈 初始性能 (Epoch {data['epochs'][0]}):")
    report.append(f"   - Accuracy: {data['accuracies'][0]:.2f}%")
    report.append(f"   - Recall: {data['recalls'][0]:.2f}%")
    report.append(f"   - Loss: {data['losses'][0]:.4f}")
    report.append("")
    
    # 最终性能
    report.append(f"🏆 最终性能 (Epoch {data['epochs'][-1]}):")
    report.append(f"   - Accuracy: {data['accuracies'][-1]:.2f}%")
    report.append(f"   - Precision: {data['precisions'][-1]:.2f}%")
    report.append(f"   - Recall: {data['recalls'][-1]:.2f}%")
    report.append(f"   - F1-Score: {data['f1_scores'][-1]:.2f}%")
    report.append(f"   - Loss: {data['losses'][-1]:.4f}")
    report.append("")
    
    # 性能提升
    acc_improvement = data['accuracies'][-1] - data['accuracies'][0]
    rec_improvement = data['recalls'][-1] - data['recalls'][0]
    report.append(f"📈 性能提升:")
    report.append(f"   - Accuracy: +{acc_improvement:.2f}%")
    report.append(f"   - Recall: +{rec_improvement:.2f}%")
    report.append("")
    
    # 最佳性能
    best_acc_idx = np.argmax(data['accuracies'])
    best_rec_idx = np.argmax(data['recalls'])
    best_f1_idx = np.argmax(data['f1_scores'])
    
    report.append(f"🌟 最佳指标:")
    report.append(f"   - 最高Accuracy: {data['accuracies'][best_acc_idx]:.2f}% (Epoch {data['epochs'][best_acc_idx]})")
    report.append(f"   - 最高Recall: {data['recalls'][best_rec_idx]:.2f}% (Epoch {data['epochs'][best_rec_idx]})")
    report.append(f"   - 最高F1-Score: {data['f1_scores'][best_f1_idx]:.2f}% (Epoch {data['epochs'][best_f1_idx]})")
    report.append("")
    
    # 混淆矩阵
    tp, fp, fn, tn = data['tps'][-1], data['fps'][-1], data['fns'][-1], data['tns'][-1]
    report.append(f"🎯 最终混淆矩阵:")
    report.append(f"   - True Positive (TP): {tp}")
    report.append(f"   - True Negative (TN): {tn}")
    report.append(f"   - False Positive (FP): {fp}")
    report.append(f"   - False Negative (FN): {fn}")
    report.append(f"   - Total Samples: {tp + tn + fp + fn}")
    report.append("")
    
    # 误报率和漏检率
    fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
    report.append(f"⚠️ 错误分析:")
    report.append(f"   - False Positive Rate (FPR): {fpr:.2f}%")
    report.append(f"   - False Negative Rate (FNR): {fnr:.2f}%")
    report.append("")
    
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    # 保存到文件
    with open(output_dir / 'training_statistics.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✅ 统计报告已保存: {output_dir / 'training_statistics.txt'}")

if __name__ == "__main__":
    log_file = "scene_level_training.log"
    output_dir = "training_visualizations"
    
    print("🚀 开始解析训练日志...")
    data = parse_training_log(log_file)
    
    if len(data['epochs']) == 0:
        print("❌ 未找到训练数据，请检查日志文件格式")
    else:
        print(f"✅ 成功解析 {len(data['epochs'])} 个epoch的数据\n")
        
        print("📊 生成可视化图表...")
        plot_training_curves(data, output_dir)
        
        print("\n📄 生成统计报告...")
        generate_statistics_report(data, output_dir)
        
        print(f"\n🎉 所有可视化完成！输出目录: {output_dir}/")
