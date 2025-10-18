#!/usr/bin/env python3
"""
训练监控脚本 - 实时查看训练进度
"""

import time
import os
import re
from datetime import datetime

def parse_log():
    """解析训练日志"""
    log_file = 'correct_training.log'
    
    if not os.path.exists(log_file):
        print("❌ 训练日志不存在")
        return
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # 提取关键信息
    device_match = re.search(r'使用设备: (\w+)', content)
    gpu_match = re.search(r'GPU: (.*?)\\n', content)
    samples_match = re.search(r'数据加载完成: (\d+) 样本', content)
    params_match = re.search(r'模型参数: ([\d,]+)', content)
    
    # 提取epoch总结
    epoch_summaries = re.findall(
        r'📊 Epoch (\d+) Summary:\s+Loss: ([\d.]+)\s+Accuracy: ([\d.]+)%\s+Avg Anomaly Score: ([\d.]+)',
        content
    )
    
    # 提取最新进度
    progress_match = re.findall(
        r'Epoch (\d+):\s+(\d+)%.*?(\d+)/(\d+).*?loss=([\d.]+).*?acc=([\d.]+)%.*?avg_score=([\d.]+)',
        content
    )
    
    print("=" * 70)
    print(f"🚀 MUVO异常检测训练监控")
    print(f"⏰ 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    if device_match:
        print(f"\n📱 设备信息:")
        print(f"   - 设备: {device_match.group(1)}")
        if gpu_match:
            print(f"   - GPU: {gpu_match.group(1)}")
    
    if samples_match:
        print(f"\n📊 数据信息:")
        print(f"   - 训练样本: {samples_match.group(1)}")
    
    if params_match:
        print(f"\n🏗️ 模型信息:")
        print(f"   - 参数量: {params_match.group(1)}")
    
    if epoch_summaries:
        print(f"\n📈 训练历史 (已完成 {len(epoch_summaries)} 个Epoch):")
        print(f"{'Epoch':<8} {'Loss':<10} {'Accuracy':<12} {'Avg Score':<12}")
        print("-" * 50)
        for epoch, loss, acc, score in epoch_summaries[-10:]:  # 显示最近10个
            print(f"{epoch:<8} {loss:<10} {acc + '%':<12} {score:<12}")
        
        # 计算趋势
        if len(epoch_summaries) >= 2:
            first_loss = float(epoch_summaries[0][1])
            last_loss = float(epoch_summaries[-1][1])
            loss_change = ((last_loss - first_loss) / first_loss) * 100
            
            first_acc = float(epoch_summaries[0][2])
            last_acc = float(epoch_summaries[-1][2])
            acc_change = last_acc - first_acc
            
            print(f"\n📉 训练趋势:")
            print(f"   - Loss变化: {loss_change:+.2f}% ({first_loss:.4f} → {last_loss:.4f})")
            print(f"   - Accuracy变化: {acc_change:+.2f}% ({first_acc:.2f}% → {last_acc:.2f}%)")
    
    if progress_match:
        latest = progress_match[-1]
        epoch, percent, current, total, loss, acc, score = latest
        print(f"\n⚡ 当前进度:")
        print(f"   - Epoch {epoch}: {percent}% ({current}/{total})")
        print(f"   - 当前Loss: {loss}")
        print(f"   - 当前Accuracy: {acc}%")
        print(f"   - 当前Anomaly Score: {score}")
        
        # 估算剩余时间
        completed_epochs = len(epoch_summaries)
        total_epochs = 30
        if completed_epochs > 0:
            remaining_epochs = total_epochs - completed_epochs
            print(f"\n⏳ 进度估算:")
            print(f"   - 已完成: {completed_epochs}/{total_epochs} epochs")
            print(f"   - 剩余: {remaining_epochs} epochs")
            if completed_epochs >= 2:
                # 假设每个epoch约25秒
                est_remaining_min = (remaining_epochs * 25) / 60
                print(f"   - 预计剩余时间: ~{est_remaining_min:.1f}分钟")
    
    # 检查是否训练完成
    if '🎉 训练完成！' in content:
        print(f"\n" + "=" * 70)
        print("✅ 训练已完成！")
        print("=" * 70)
        
        # 提取最终结果
        final_result = re.search(
            r'📈 最终结果:.*?最终Loss: ([\d.]+).*?最终Accuracy: ([\d.]+)%',
            content,
            re.DOTALL
        )
        if final_result:
            print(f"\n🎯 最终成绩:")
            print(f"   - Loss: {final_result.group(1)}")
            print(f"   - Accuracy: {final_result.group(2)}%")
        
        print(f"\n📁 输出文件:")
        if os.path.exists('checkpoints/real_anomaly_detector.pth'):
            size_mb = os.path.getsize('checkpoints/real_anomaly_detector.pth') / 1024 / 1024
            print(f"   ✅ 模型: checkpoints/real_anomaly_detector.pth ({size_mb:.1f} MB)")
        if os.path.exists('visualizations/real_training_results.png'):
            print(f"   ✅ 可视化: visualizations/real_training_results.png")
    else:
        print(f"\n💡 提示: 训练正在进行中，使用 'python monitor_training.py' 查看实时进度")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    try:
        parse_log()
    except KeyboardInterrupt:
        print("\n\n👋 监控已停止")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

