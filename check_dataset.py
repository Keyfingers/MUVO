#!/usr/bin/env python3
"""
数据集检查脚本
快速验证AnoVox数据集是否正确配置
"""

import os
import sys

def check_dataset():
    """检查数据集配置"""
    
    print("=" * 60)
    print("🔍 AnoVox数据集配置检查")
    print("=" * 60)
    
    # 数据集路径
    dataset_path = '/root/autodl-tmp/datasets/AnoVox'
    
    print(f"\n📁 数据集路径: {dataset_path}")
    
    # 检查路径是否存在
    if not os.path.exists(dataset_path):
        print(f"❌ 错误: 数据集路径不存在!")
        print(f"\n💡 解决方案:")
        print(f"   1. 创建目录: mkdir -p {dataset_path}")
        print(f"   2. 上传AnoVox数据集到该目录")
        print(f"   3. 查看配置指南: cat AnoVox数据集配置指南.md")
        return False
    
    print(f"✅ 数据集路径存在")
    
    # 检查目录内容
    print(f"\n📂 目录内容:")
    try:
        items = os.listdir(dataset_path)
        if not items:
            print(f"   ⚠️  目录为空，需要上传数据集")
            print(f"\n💡 数据集应包含:")
            print(f"   - trainval/ 目录")
            print(f"     - train/ (训练数据)")
            print(f"     - val/ (验证数据)")
            return False
        
        for item in sorted(items):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                # 统计子目录数量
                try:
                    sub_items = os.listdir(item_path)
                    print(f"   📁 {item}/ ({len(sub_items)} 项)")
                except:
                    print(f"   📁 {item}/")
            else:
                # 显示文件大小
                size = os.path.getsize(item_path)
                size_mb = size / (1024 * 1024)
                print(f"   📄 {item} ({size_mb:.2f} MB)")
        
        print(f"\n✅ 找到 {len(items)} 个项目")
        
    except Exception as e:
        print(f"❌ 错误: 无法读取目录内容: {e}")
        return False
    
    # 检查必需的目录结构
    print(f"\n🔍 检查数据集结构:")
    
    required_paths = [
        'trainval',
        'trainval/train',
    ]
    
    optional_paths = [
        'trainval/val',
        'trainval/test',
    ]
    
    all_good = True
    
    for path in required_paths:
        full_path = os.path.join(dataset_path, path)
        if os.path.exists(full_path):
            print(f"   ✅ {path}/")
        else:
            print(f"   ❌ {path}/ (必需)")
            all_good = False
    
    for path in optional_paths:
        full_path = os.path.join(dataset_path, path)
        if os.path.exists(full_path):
            print(f"   ✅ {path}/ (可选)")
        else:
            print(f"   ⚠️  {path}/ (可选，未找到)")
    
    # 检查训练数据
    train_path = os.path.join(dataset_path, 'trainval', 'train')
    if os.path.exists(train_path):
        print(f"\n📊 训练数据统计:")
        try:
            scenes = os.listdir(train_path)
            print(f"   场景数量: {len(scenes)}")
            
            total_episodes = 0
            for scene in scenes[:3]:  # 只检查前3个场景
                scene_path = os.path.join(train_path, scene)
                if os.path.isdir(scene_path):
                    episodes = [d for d in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, d))]
                    print(f"   - {scene}: {len(episodes)} 个episode")
                    total_episodes += len(episodes)
            
            if len(scenes) > 3:
                print(f"   ... (还有 {len(scenes) - 3} 个场景)")
            
            print(f"   总计检查: {total_episodes} 个episode")
            
        except Exception as e:
            print(f"   ⚠️  无法统计: {e}")
    
    # 检查配置文件
    print(f"\n⚙️  检查配置文件:")
    config_path = 'muvo/configs/anomaly_detection.yml'
    if os.path.exists(config_path):
        print(f"   ✅ {config_path} 存在")
        
        # 检查配置内容
        try:
            with open(config_path, 'r') as f:
                content = f.read()
                if dataset_path in content:
                    print(f"   ✅ DATAROOT 已正确配置")
                else:
                    print(f"   ⚠️  DATAROOT 可能需要更新")
        except:
            pass
    else:
        print(f"   ❌ {config_path} 不存在")
        all_good = False
    
    # 总结
    print(f"\n" + "=" * 60)
    if all_good and items:
        print("✅ 数据集配置检查通过!")
        print("🚀 可以开始训练了!")
        print(f"\n运行训练命令:")
        print(f"  python train_anomaly_detection.py --config-file muvo/configs/anomaly_detection.yml")
    elif not items:
        print("⚠️  数据集目录为空")
        print("📖 请查看: AnoVox数据集配置指南.md")
        print(f"\n快速上传命令示例:")
        print(f"  # 从本地上传")
        print(f"  scp -r /path/to/AnoVox root@your-server:{dataset_path}")
    else:
        print("⚠️  数据集配置不完整")
        print("📖 请查看: AnoVox数据集配置指南.md")
    
    print("=" * 60)
    
    return all_good and bool(items)


if __name__ == '__main__':
    try:
        success = check_dataset()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  检查被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

