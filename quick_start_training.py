"""
快速开始训练脚本 - 简化版
用于快速测试训练流程
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 导入数据加载器
from muvo.dataset.anovox_dataset import create_anovox_dataloader


class SimpleAnomalyDetector(nn.Module):
    """
    简化的异常检测模型 - 用于快速测试
    """
    def __init__(self):
        super().__init__()
        
        # 图像特征提取（简化）
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # 点云特征提取（简化 - 使用PointNet风格）
        self.point_mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(64 * 64 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, batch):
        # 处理图像
        img = batch['image']  # [B, 3, H, W]
        img_feat = self.image_conv(img)  # [B, 64, 8, 8]
        img_feat = img_feat.view(img_feat.size(0), -1)  # [B, 64*64]
        
        # 处理点云（简化：取最大池化）
        points = batch['points']  # [B, N, 4]
        # 限制点数以节省显存
        if points.size(1) > 10000:
            indices = torch.randperm(points.size(1))[:10000]
            points = points[:, indices, :]
        
        point_feat = self.point_mlp(points)  # [B, N, 128]
        point_feat = torch.max(point_feat, dim=1)[0]  # [B, 128] 全局最大池化
        
        # 融合
        combined = torch.cat([img_feat, point_feat], dim=1)  # [B, 64*64+128]
        anomaly_score = self.fusion(combined)  # [B, 1]
        
        return {
            'anomaly_score': anomaly_score,
            'image_features': img_feat,
            'point_features': point_feat
        }


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_anomaly = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # 移动数据到设备
        batch['image'] = batch['image'].to(device)
        batch['points'] = batch['points'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(batch)
        
        # 计算损失（自监督：假设大部分是正常的）
        anomaly_score = outputs['anomaly_score']
        # 简单损失：鼓励分数接近0（正常）+ 一些正则化
        loss = torch.mean(anomaly_score) + 0.1 * torch.std(anomaly_score)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        total_anomaly += anomaly_score.mean().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'anomaly': f'{anomaly_score.mean().item():.4f}'
        })
        
        # 完整训练：不限制步数
        # if batch_idx >= 100:
        #     break
    
    avg_loss = total_loss / len(dataloader)
    avg_anomaly = total_anomaly / len(dataloader)
    
    return avg_loss, avg_anomaly


def visualize_results(train_losses, train_anomalies, save_dir='visualizations'):
    """可视化训练结果"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses, 'b-', linewidth=2, marker='o')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 异常分数曲线
    ax2.plot(train_anomalies, 'r-', linewidth=2, marker='s')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Anomaly Score', fontsize=12)
    ax2.set_title('Average Anomaly Score', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / 'quick_training_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化保存至: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("🚀 快速训练测试开始")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 创建数据加载器
    print(f"\n📁 加载数据...")
    train_loader = create_anovox_dataloader(
        data_root='/root/autodl-tmp/datasets/AnoVox/AnoVox_Dynamic_Mono_Town07',
        split='train',
        batch_size=2,  # 小batch size
        num_workers=2,
        shuffle=True
    )
    print(f"✅ 数据加载完成: {len(train_loader.dataset)} 样本")
    
    # 创建模型
    print(f"\n🏗️ 初始化模型...")
    model = SimpleAnomalyDetector().to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型参数: {total_params:,}")
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # 训练配置
    num_epochs = 50  # 完整训练
    
    # 训练循环
    print(f"\n🎓 开始完整训练...")
    print(f"💡 训练{num_epochs}个epochs，使用全部数据\n")
    train_losses = []
    train_anomalies = []
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        loss, anomaly = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        train_losses.append(loss)
        train_anomalies.append(anomaly)
        
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Loss: {loss:.4f}")
        print(f"   Anomaly Score: {anomaly:.4f}")
        print(f"   Time: {time.time() - start_time:.1f}s\n")
    
    # 保存模型
    model_path = Path('checkpoints') / 'simple_anomaly_detector.pth'
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"💾 模型已保存: {model_path}")
    
    # 可视化结果
    print(f"\n📊 生成可视化...")
    visualize_results(train_losses, train_anomalies)
    
    print("\n" + "=" * 60)
    print("🎉 快速训练测试完成！")
    print("=" * 60)
    print(f"\n📈 训练总结:")
    print(f"   - 总时间: {time.time() - start_time:.1f}秒")
    print(f"   - 初始损失: {train_losses[0]:.4f}")
    print(f"   - 最终损失: {train_losses[-1]:.4f}")
    print(f"   - 改善: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%")
    print(f"\n📁 输出文件:")
    print(f"   - 模型: checkpoints/simple_anomaly_detector.pth")
    print(f"   - 可视化: visualizations/quick_training_results.png")


if __name__ == '__main__':
    main()

