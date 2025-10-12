# AnoVox 数据集配置指南

## 📍 数据集放置位置

### 推荐目录：
```
/root/autodl-tmp/datasets/AnoVox/
```

已经为您创建好了！✅

## 📂 AnoVox数据集应有的结构

根据项目需求，您的AnoVox数据集应该遵循以下结构：

```
/root/autodl-tmp/datasets/AnoVox/
├── trainval/
│   ├── train/
│   │   ├── Town01/          # 或者AnoVox特定的场景名称
│   │   │   ├── 0000/
│   │   │   │   ├── image/
│   │   │   │   │   ├── image_000000000.png
│   │   │   │   │   ├── image_000000001.png
│   │   │   │   │   └── ...
│   │   │   │   ├── points/  # 点云数据
│   │   │   │   │   ├── points_000000000.npy
│   │   │   │   │   └── ...
│   │   │   │   ├── voxel/   # 体素数据（用于异常检测）
│   │   │   │   │   ├── voxel_000000000.npy
│   │   │   │   │   └── ...
│   │   │   │   ├── range_view_pcd_xyzd/  # range-view点云
│   │   │   │   │   └── ...
│   │   │   │   └── pd_dataframe.pkl  # 元数据
│   │   │   ├── 0001/
│   │   │   └── ...
│   │   ├── Town02/  # 或其他场景
│   │   └── ...
│   ├── val/
│   │   └── (同样的结构)
│   └── test/
│       └── (同样的结构)
```

## 🔧 配置步骤

### 1. 上传数据集到服务器

您可以使用以下方式上传AnoVox数据集：

#### 方式A：从本地上传
```bash
# 在本地终端执行
scp -r /path/to/AnoVox root@your-server:/root/autodl-tmp/datasets/
```

#### 方式B：从网盘/云存储下载
```bash
# 在服务器上执行
cd /root/autodl-tmp/datasets/AnoVox
# 使用wget、curl或其他下载工具
wget your_dataset_url
# 解压
unzip AnoVox.zip  # 或 tar -xzf AnoVox.tar.gz
```

#### 方式C：使用autodl数据集功能
```bash
# 如果数据集已经在autodl平台上
# 可以直接挂载或复制
```

### 2. 修改配置文件

将数据集路径配置到训练配置中：

**编辑 `muvo/configs/anomaly_detection.yml`**：
```yaml
DATASET:
  DATAROOT: '/root/autodl-tmp/datasets/AnoVox'  # 修改这里
  VERSION: 'trainval'
  STRIDE_SEC: 0.2
  FILTER_BEGINNING_OF_RUN_SEC: 1.0
  FILTER_NORM_REWARD: 0.6
```

### 3. 验证数据集

创建一个简单的验证脚本：

```bash
cd /root/autodl-tmp/MUVO/MUVO
python -c "
import os
dataset_path = '/root/autodl-tmp/datasets/AnoVox'
print(f'检查数据集路径: {dataset_path}')
print(f'路径存在: {os.path.exists(dataset_path)}')
if os.path.exists(dataset_path):
    print('目录内容:')
    for item in os.listdir(dataset_path):
        print(f'  - {item}')
"
```

## 📋 AnoVox数据集关键文件说明

### 必需的数据类型：

1. **图像数据** (`image/`)
   - RGB图像，用于视觉特征提取
   - 格式：PNG或JPG
   - 推荐尺寸：600×960（可配置）

2. **点云数据** (`points/` 或 `range_view_pcd_xyzd/`)
   - 3D点云，用于空间特征提取
   - 格式：NPY或BIN
   - 包含XYZ坐标和强度信息

3. **体素数据** (`voxel/`)
   - 3D体素网格，用于异常检测
   - 格式：NPY
   - 默认尺寸：192×192×64（可配置）

4. **元数据** (`pd_dataframe.pkl`)
   - 包含时间戳、传感器参数等
   - 格式：Pickle

### 可选数据：
- `birdview/` - 鸟瞰图
- `depth_semantic/` - 深度语义图
- `routemap/` - 路径图

## 🚀 启动训练

数据集配置完成后，运行：

```bash
cd /root/autodl-tmp/MUVO/MUVO

# 方式1：使用异常检测配置
python train_anomaly_detection.py --config-file muvo/configs/anomaly_detection.yml

# 方式2：使用原始训练脚本
python train.py --config-file muvo/configs/anomaly_detection.yml
```

## 🔍 数据集检查清单

在开始训练前，请确认：

- [ ] 数据集已上传到 `/root/autodl-tmp/datasets/AnoVox`
- [ ] 目录结构符合要求（trainval/train/, trainval/val/）
- [ ] 包含必需的数据类型（image, points, voxel）
- [ ] 配置文件中的DATAROOT已正确设置
- [ ] 运行验证脚本确认数据可访问

## 💡 常见问题

### Q1: 数据集很大，上传很慢怎么办？
A: 可以考虑：
- 使用rsync增量上传
- 压缩后上传，在服务器上解压
- 分批上传，先用小部分数据测试

### Q2: 数据集格式与MUVO不完全匹配？
A: 需要编写数据预处理脚本转换格式，或修改dataloader

### Q3: 如何验证数据集加载正确？
A: 运行测试脚本或在训练前打印一个batch的数据

## 📞 需要帮助？

如果您在配置数据集时遇到问题：
1. 检查数据集目录权限
2. 查看数据集文件结构是否正确
3. 运行验证脚本检查数据可读性
4. 查看训练日志的数据加载错误信息

---

**准备好数据集后，您就可以开始训练您的跨模态异常检测模型了！** 🎯

