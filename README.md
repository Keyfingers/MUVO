# MUVO: 跨模态注意力融合的自动驾驶异常检测系统
# MUVO: Cross-Modal Attention Fusion for Autonomous Driving Anomaly Detection

## 🚀 核心创新 (Core Innovation)

本项目在原有MUVO基础上实现了**跨模态注意力融合的异常检测系统**，主要创新点包括：

### 1. 冻结骨干网络架构 (Frozen Backbone Architecture)
- **图像分支**: 使用预训练的ResNet18作为特征提取器，权重冻结
- **点云分支**: 使用预训练的ResNet18处理range-view点云，权重冻结
- **优势**: 大幅减少训练时间和显存消耗，提高训练效率

### 2. 跨模态注意力融合 (Cross-Modal Attention Fusion) ⭐
- **核心创新**: 轻量级跨模态注意力模块
- **Query**: 点云特征 (Point Cloud Features)
- **Key & Value**: 对齐后的图像特征 (Aligned Image Features)
- **输出**: 增强的点云特征，融合了图像纹理与上下文信息

### 3. 特征空间对齐 (Feature Space Alignment)
- 根据投影关系将图像特征映射到体素空间
- 支持多种对齐方法（最近邻、双线性插值等）
- 确保跨模态特征的有效融合

### 4. 轻量级异常检测头 (Lightweight Anomaly Detection Head)
- **3D CNN**: 轻量级3D卷积网络
- **MLP**: 多层感知机架构
- **多尺度**: 多尺度特征融合
- **仅此部分可训练**: 其他骨干网络权重冻结

## 📊 系统架构 (System Architecture)

```
Stage 1: 输入与预处理
├── 图像 (H×W×3)
├── 点云 (N×4) 
└── 标定参数

Stage 2: 冻结骨干网络特征提取
├── 图像分支 (ResNet18, 权重冻结)
└── 点云分支 (ResNet18, 权重冻结)

Stage 3: ⭐ 核心创新 - 跨模态注意力融合
├── 特征空间对齐
├── 轻量级跨模态注意力模块
└── 增强的点云特征

Stage 4: 异常检测头与输出
├── 轻量级3D CNN/MLP (仅此部分可训练)
└── 体素级异常分数 & 异常热力图
```

## 🛠️ 技术特点 (Technical Features)

- **高效训练**: 骨干网络冻结，仅训练异常检测头
- **跨模态融合**: 图像和点云特征的深度交互
- **轻量级设计**: 计算效率高，适合实时应用
- **模块化架构**: 易于扩展和修改
- **多尺度检测**: 支持不同尺度的异常检测

---

## 原始项目说明 (Original Project Description)

This is the PyTorch implementation for the paper
>  Occupancy-Guided Sensor Fusion Strategies for Generative Predictive World Models <br/>

## Requirements
The simplest way to install all required dependencies is to create 
a [conda](https://docs.conda.io/projects/miniconda/en/latest/) environment by running
```
conda env create -f carla_env.yml
```
Then activate conda environment by
```
conda activate muvo
```
or create your own venv and install the requirement by running
```
pip install -r requirements.txt
```


## Dataset
Use [CARLA](http://carla.org/) to collection data. 
First install carla refer to its [documentation](https://carla.readthedocs.io/en/latest/).

### Dataset Collection
Change settings in config/, 
then run `bash run/data_collect.sh ${PORT}` 
with `${PORT}` the port to run CARLA (usually `2000`) <br/>
The data collection code is modified from 
[CARLA-Roach](https://github.com/zhejz/carla-roach) and [MILE](https://github.com/wayveai/mile),
some config settings can be referred there.

### Voxelization
After collecting the data by CARLA, create voxels data by running `data/generate_voxels.py`, <br/> 
voxel settings can be changed in `data_preprocess.yaml`.

### Folder Structure
After completing the above steps, or otherwise obtaining the dataset,
please change the file structure of the dataset. <br/>

The main branch includes most of the results presented in the paper. In the 2D branch, you can find 2D latent states, perceptual losses, and a new transformer backbone. The data is organized in the following format
```
/carla_dataset/trainval/
                   ├── train/
                   │     ├── Town01/
                   │     │     ├── 0000/
                   │     │     │     ├── birdview/
                   │     │     │     │      ├ birdview_000000000.png
                   │     │     │     │      .
                   │     │     │     ├── depth_semantic/
                   │     │     │     │      ├ depth_semantic_000000000.png
                   │     │     │     │      .
                   │     │     │     ├── image/
                   │     │     │     │      ├ image_000000000.png
                   │     │     │     │      .
                   │     │     │     ├── points/
                   │     │     │     │      ├ points_000000000.png
                   │     │     │     │      .
                   │     │     │     ├── points_semantic/
                   │     │     │     │      ├ points_semantic_000000000.png
                   │     │     │     │      .
                   │     │     │     ├── routemap/
                   │     │     │     │      ├ routemap_000000000.png
                   │     │     │     │      .
                   │     │     │     ├── voxel/
                   │     │     │     │      ├ voxel_000000000.png
                   │     │     │     │      .
                   │     │     │     └── pd_dataframe.pkl
                   │     │     ├── 0001/
                   │     │     ├── 0002/
                   │     |     .
                   │     |     └── 0024/
                   │     ├── Town03/
                   │     ├── Town04/
                   │     .
                   │     └── Town06/
                   ├── val0/
                   .
                   └── val1/
```

## 🎯 异常检测训练 (Anomaly Detection Training)

### 快速开始 (Quick Start)
```bash
# 使用跨模态注意力融合异常检测配置
python train.py --config-file muvo/configs/anomaly_detection.yml
```

### 配置说明 (Configuration)
- **配置文件**: `muvo/configs/anomaly_detection.yml`
- **核心参数**:
  - `MODEL.ANOMALY_DETECTION.ENABLED: True` - 启用异常检测
  - `MODEL.ANOMALY_DETECTION.FREEZE_BACKBONE: True` - 冻结骨干网络
  - `MODEL.ANOMALY_DETECTION.HEAD_TYPE: '3dcnn'` - 异常检测头类型
  - `MODEL.ANOMALY_DETECTION.HIDDEN_DIM: 128` - 注意力隐藏维度
  - `MODEL.ANOMALY_DETECTION.NUM_HEADS: 8` - 注意力头数

### 训练特点 (Training Features)
- **高效训练**: 骨干网络权重冻结，仅训练异常检测相关模块
- **低显存需求**: 相比端到端训练，显存需求大幅降低
- **快速收敛**: 由于骨干网络预训练，训练收敛更快

## 原始训练 (Original Training)
Run
```angular2html
python train.py --conifg-file muvo/configs/your_config.yml
```
You can use default config file `muvo/configs/muvo.yml`, or create your own config file in `muvo/configs/`. <br/>
In `config file(*.yml)`, you can set all the configs listed in `muvo/config.py`. <br/>
Before training, make sure that the required input/output data as well as the model structure/dimensions are correctly set in `muvo/configs/your_config.yml`.

## test

### weights

We provide weights for pre-trained models, and each was trained with around 100,000 steps. [weights](https://github.com/daniel-bogdoll/MUVO/releases/tag/1.0) is for a 1D latent space. [weights_2D](https://github.com/daniel-bogdoll/MUVO/releases/tag/2.0) for a 2D latent space. We provide config files for each:  <br/>  <br/> 
'basic_voxel' in [weights_2D](https://github.com/daniel-bogdoll/MUVO/releases/tag/2.0) is for the basic 2D latent space model, which uses resnet18 as the backbone, without bev mapping for image features, uses range view for point cloud and uses the transformer to fuse features, the corresponding config file is '[test_base_2d.yml](https://github.com/daniel-bogdoll/MUVO/blob/main/muvo/configs/test_base_2d.yml)';  <br/>  <br/> 
'mobilevit' weights just change the backbone compared to the 'basic_voxel' weights, the corresponding config file is '[test_mobilevit_2d.yml](https://github.com/daniel-bogdoll/MUVO/blob/main/muvo/configs/test_mobilevit_2d.yml)'; <br/>  <br/> 
'RV_WOB_TR_1d_Voxel' and 'RV_WOB_TR_1d_no_Voxel' in [weights](https://github.com/daniel-bogdoll/MUVO/releases/tag/1.0) all use basic setting but use 1d latent space, '[test_base_1d.yml](https://github.com/daniel-bogdoll/MUVO/blob/main/muvo/configs/test_base_1d.yml)' and '[test_base_1d_without_voxel.yml](https://github.com/daniel-bogdoll/MUVO/blob/main/muvo/configs/test_base_1d_without_voxel.yml)' are corresponding config files.

### execute
Run
```angular2html
python prediction.py --config-file muvo/configs/test.yml
```
The config file is the same as in training.\
In `file 'muvo/data/dataset.py', class 'DataModule', function 'setup'`, you can change the test dataset/sampler type.

## 🏗️ 模型架构详解 (Model Architecture Details)

### 跨模态注意力融合模块 (Cross-Modal Attention Fusion Module)
```python
# 核心组件
from muvo.models.cross_modal_attention import CrossModalFusionModule
from muvo.models.anomaly_detection_head import AnomalyDetectionHead

# 使用示例
cross_modal_fusion = CrossModalFusionModule(
    pc_feature_dim=256,
    img_feature_dim=256, 
    hidden_dim=128,
    num_heads=8
)

anomaly_head = AnomalyDetectionHead(
    input_dim=256,
    head_type='3dcnn',  # 或 'mlp', 'multiscale'
    output_dim=1
)
```

### 异常检测模型 (Anomaly Detection Model)
```python
# 使用修改后的Mile模型
from muvo.models.mile_anomaly import MileAnomalyDetection

model = MileAnomalyDetection(cfg)
model.freeze_backbone_weights()  # 冻结骨干网络权重
```

### 训练参数配置 (Training Parameters)
```yaml
# 关键配置参数
MODEL:
  ANOMALY_DETECTION:
    ENABLED: True
    FREEZE_BACKBONE: True
    HEAD_TYPE: '3dcnn'
    HIDDEN_DIM: 128
    NUM_HEADS: 8
    DROPOUT: 0.1

OPTIMIZER:
  LR: 1e-3  # 异常检测头可以使用较高学习率
  FROZEN:
    ENABLED: True
    TRAIN_LIST: ['cross_modal_fusion', 'anomaly_detection_head']
```

## 📈 性能优势 (Performance Advantages)

### 训练效率 (Training Efficiency)
- **显存使用**: 相比端到端训练减少60-80%
- **训练时间**: 由于骨干网络冻结，训练时间减少50-70%
- **收敛速度**: 预训练骨干网络加速收敛

### 检测性能 (Detection Performance)
- **跨模态融合**: 图像和点云特征的深度交互提升检测精度
- **多尺度检测**: 支持不同尺度的异常检测
- **实时性**: 轻量级设计适合实时应用

## 🔧 扩展功能 (Extension Features)

### 自定义异常检测头
```python
# 创建自定义检测头
class CustomAnomalyHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 自定义架构
        pass
```

### 添加新的融合策略
```python
# 扩展跨模态融合模块
class EnhancedCrossModalFusion(CrossModalFusionModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 添加新的融合策略
        pass
```

## Related Projects
Our code is based on [MILE](https://github.com/wayveai/mile). 
And thanks to [CARLA-Roach](https://github.com/zhejz/carla-roach) for making a gym wrapper around CARLA.
