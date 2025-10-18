"""
训练可视化脚本
实时监控训练进度和生成可视化结果
"""

import os
import time
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def plot_training_curves(log_dir, save_dir='visualizations'):
    """
    绘制训练曲线
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 查找最新的事件文件
    log_path = Path(log_dir)
    event_files = list(log_path.rglob('events.out.tfevents.*'))
    
    if not event_files:
        print(f"⚠️ 未找到TensorBoard日志文件: {log_dir}")
        return
    
    # 使用最新的事件文件
    latest_event = max(event_files, key=lambda p: p.stat().st_mtime)
    print(f"📊 读取日志: {latest_event}")
    
    # 加载事件
    ea = event_accumulator.EventAccumulator(str(latest_event.parent))
    ea.Reload()
    
    # 获取可用的标量
    scalar_tags = ea.Tags()['scalars']
    print(f"✅ 可用指标: {scalar_tags}")
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('AnoVox Anomaly Detection Training Progress', fontsize=16, fontweight='bold')
    
    # 1. 训练和验证损失
    ax = axes[0, 0]
    if 'train_loss_epoch' in scalar_tags:
        train_loss = [(s.step, s.value) for s in ea.Scalars('train_loss_epoch')]
        steps, values = zip(*train_loss)
        ax.plot(steps, values, 'b-', label='Train Loss', linewidth=2)
    
    if 'val_loss' in scalar_tags:
        val_loss = [(s.step, s.value) for s in ea.Scalars('val_loss')]
        steps, values = zip(*val_loss)
        ax.plot(steps, values, 'r-', label='Val Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. 异常概率
    ax = axes[0, 1]
    if 'train_anomaly_prob' in scalar_tags:
        train_prob = [(s.step, s.value) for s in ea.Scalars('train_anomaly_prob')]
        steps, values = zip(*train_prob)
        ax.plot(steps, values, 'g-', label='Train Anomaly Prob', linewidth=2)
    
    if 'val_anomaly_prob' in scalar_tags:
        val_prob = [(s.step, s.value) for s in ea.Scalars('val_anomaly_prob')]
        steps, values = zip(*val_prob)
        ax.plot(steps, values, 'm-', label='Val Anomaly Prob', linewidth=2)
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Anomaly Probability', fontsize=12)
    ax.set_title('Anomaly Detection Probability', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. 学习率
    ax = axes[1, 0]
    if 'lr-AdamW' in scalar_tags:
        lr_data = [(s.step, s.value) for s in ea.Scalars('lr-AdamW')]
        steps, values = zip(*lr_data)
        ax.plot(steps, values, 'orange', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # 4. 训练步损失（详细）
    ax = axes[1, 1]
    if 'train_loss_step' in scalar_tags:
        train_loss_step = [(s.step, s.value) for s in ea.Scalars('train_loss_step')]
        steps, values = zip(*train_loss_step)
        # 使用移动平均平滑曲线
        window = min(50, len(values) // 10)
        if window > 0:
            smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
            ax.plot(steps[:len(smoothed)], smoothed, 'b-', linewidth=2, alpha=0.7)
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss (Smoothed)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    save_path = save_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 训练曲线保存至: {save_path}")
    plt.close()
    
    # 生成统计报告
    generate_stats_report(ea, scalar_tags, save_dir)


def generate_stats_report(ea, scalar_tags, save_dir):
    """生成统计报告"""
    report_path = save_dir / 'training_stats.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("AnoVox异常检测训练统计报告\n")
        f.write("=" * 60 + "\n\n")
        
        # 训练损失统计
        if 'train_loss_epoch' in scalar_tags:
            train_loss = [s.value for s in ea.Scalars('train_loss_epoch')]
            f.write(f"📊 训练损失 (Train Loss):\n")
            f.write(f"   - 初始: {train_loss[0]:.4f}\n")
            f.write(f"   - 最终: {train_loss[-1]:.4f}\n")
            f.write(f"   - 最小: {min(train_loss):.4f}\n")
            f.write(f"   - 平均: {np.mean(train_loss):.4f}\n")
            f.write(f"   - 改善: {((train_loss[0] - train_loss[-1]) / train_loss[0] * 100):.1f}%\n\n")
        
        # 验证损失统计
        if 'val_loss' in scalar_tags:
            val_loss = [s.value for s in ea.Scalars('val_loss')]
            f.write(f"📊 验证损失 (Validation Loss):\n")
            f.write(f"   - 初始: {val_loss[0]:.4f}\n")
            f.write(f"   - 最终: {val_loss[-1]:.4f}\n")
            f.write(f"   - 最小: {min(val_loss):.4f}\n")
            f.write(f"   - 平均: {np.mean(val_loss):.4f}\n")
            f.write(f"   - 改善: {((val_loss[0] - val_loss[-1]) / val_loss[0] * 100):.1f}%\n\n")
        
        # 异常概率统计
        if 'train_anomaly_prob' in scalar_tags:
            train_prob = [s.value for s in ea.Scalars('train_anomaly_prob')]
            f.write(f"📊 异常概率 (Anomaly Probability):\n")
            f.write(f"   - 训练集平均: {np.mean(train_prob):.4f}\n")
        
        if 'val_anomaly_prob' in scalar_tags:
            val_prob = [s.value for s in ea.Scalars('val_anomaly_prob')]
            f.write(f"   - 验证集平均: {np.mean(val_prob):.4f}\n\n")
        
        f.write("=" * 60 + "\n")
    
    print(f"✅ 统计报告保存至: {report_path}")
    
    # 打印到控制台
    with open(report_path, 'r', encoding='utf-8') as f:
        print(f.read())


def create_visualization_grid(vis_dir='visualizations', output_path='visualizations/results_grid.png'):
    """
    创建可视化网格 - 展示多个epoch的异常热力图
    """
    vis_dir = Path(vis_dir)
    output_path = Path(output_path)
    
    # 查找所有热力图
    heatmap_files = sorted(vis_dir.glob('heatmap_epoch_*.png'))
    
    if not heatmap_files:
        print(f"⚠️ 未找到热力图文件")
        return
    
    # 选择几个关键epoch显示
    num_display = min(9, len(heatmap_files))
    indices = np.linspace(0, len(heatmap_files)-1, num_display, dtype=int)
    selected_files = [heatmap_files[i] for i in indices]
    
    # 创建网格
    rows = int(np.ceil(np.sqrt(num_display)))
    cols = int(np.ceil(num_display / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    fig.suptitle('Anomaly Detection Progress Across Epochs', fontsize=16, fontweight='bold')
    
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    for idx, (ax, img_path) in enumerate(zip(axes.flat, selected_files)):
        # 读取图像
        img = plt.imread(img_path)
        ax.imshow(img)
        
        # 从文件名提取epoch
        epoch = int(img_path.stem.split('_')[-1])
        ax.set_title(f'Epoch {epoch}', fontsize=12)
        ax.axis('off')
    
    # 隐藏多余的子图
    for idx in range(len(selected_files), rows * cols):
        axes.flat[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化网格保存至: {output_path}")
    plt.close()


def monitor_training(log_dir='lightning_logs', interval=60):
    """
    持续监控训练进度
    
    Args:
        log_dir: TensorBoard日志目录
        interval: 更新间隔（秒）
    """
    print("=" * 60)
    print("🔍 开始监控训练进度")
    print(f"📁 日志目录: {log_dir}")
    print(f"⏰ 更新间隔: {interval}秒")
    print("=" * 60)
    print("\n💡 按 Ctrl+C 停止监控\n")
    
    try:
        while True:
            # 查找最新的版本目录
            log_path = Path(log_dir) / 'anovox_anomaly'
            if not log_path.exists():
                print(f"⚠️ 等待训练开始... ({log_path})")
                time.sleep(interval)
                continue
            
            # 获取最新版本
            versions = sorted([d for d in log_path.iterdir() if d.is_dir() and d.name.startswith('version_')])
            if not versions:
                print(f"⚠️ 等待训练开始...")
                time.sleep(interval)
                continue
            
            latest_version = versions[-1]
            
            # 更新可视化
            print(f"\n🔄 更新可视化... ({time.strftime('%H:%M:%S')})")
            plot_training_curves(latest_version)
            create_visualization_grid()
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 监控已停止")


def main():
    parser = argparse.ArgumentParser(description='Visualize Training Progress')
    parser.add_argument('--log-dir', type=str, default='lightning_logs',
                       help='TensorBoard log directory')
    parser.add_argument('--monitor', action='store_true',
                       help='Continuously monitor training')
    parser.add_argument('--interval', type=int, default=60,
                       help='Update interval in seconds (for monitor mode)')
    
    args = parser.parse_args()
    
    if args.monitor:
        # 持续监控模式
        monitor_training(args.log_dir, args.interval)
    else:
        # 单次可视化模式
        log_path = Path(args.log_dir) / 'anovox_anomaly'
        if log_path.exists():
            versions = sorted([d for d in log_path.iterdir() if d.is_dir() and d.name.startswith('version_')])
            if versions:
                latest_version = versions[-1]
                print(f"📊 生成可视化: {latest_version}")
                plot_training_curves(latest_version)
                create_visualization_grid()
            else:
                print(f"⚠️ 未找到训练版本")
        else:
            print(f"⚠️ 日志目录不存在: {log_path}")


if __name__ == '__main__':
    main()

