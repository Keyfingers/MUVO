# 方案A核心代码清单

**用途**: 与导师讨论时快速定位关键代码

---

## 📁 核心文件列表

### 1. 坐标映射实现（核心！）

**文件**: `precise_voxel_mapping_v2.py` (163行)

**关键函数**: `create_precise_point_labels_from_voxels()`

**问题位置**: 第42-51行

```python
# 当前错误的实现
def create_precise_point_labels_from_voxels(...):
    # ...
    
    # ❌ 第42-51行：错误的坐标系推导
    resolution = cfg.VOXEL.RESOLUTION  # 0.2
    grid_size = cfg.VOXEL.SIZE  # [192, 192, 64]
    offset_forward = cfg.BEV.OFFSET_FORWARD  # -64
    
    grid_origin_x = -((grid_size[0] / 2) + offset_forward) * resolution  # -6.4
    grid_origin_y = -(grid_size[1] / 2) * resolution  # -19.2
    grid_origin_z = -2.0  # 假设值
    
    grid_origin = torch.tensor(
        [grid_origin_x, grid_origin_y, grid_origin_z],
        device=device
    ).view(1, 1, 3)
```

**需要修改**: 这3行计算grid_origin的代码

---

### 2. 配置文件

**文件**: `muvo/config.py` (393行)

**关键参数**: 

```python
# 第103-105行：体素配置
_C.VOXEL = CN()
_C.VOXEL.SIZE = [192, 192, 64]          # ❓ 需要明确这是什么SIZE
_C.VOXEL.RESOLUTION = 0.2                # ✅ 确认无误
_C.VOXEL.EV_POSITION = [32, 96, 12]     # ❓ 关键！可能是车辆位置

# 第138行：BEV配置
_C.BEV.OFFSET_FORWARD = -64              # ❓ 需要明确含义
```

**需要明确**: 
- `VOXEL.SIZE` 是局部网格还是全局网格？
- `EV_POSITION` 是什么？如果是车辆位置，对应的真实坐标是？

---

### 3. 数据集加载

**文件**: `muvo/dataset/anovox_dataset.py` (402行)

**体素加载**: 第209-226行

```python
# 第213-216行：✅ 这部分工作正常
voxel_data = np.load(sample_info['voxel_path'])
if isinstance(voxel_data, np.ndarray):
    voxel = voxel_data  # [N, 4] (vx, vy, vz, semantic_id)
```

**体素数据格式**: ✅ 已确认
- 形状: `[N, 4]`
- 列0-2: 体素索引 (vx, vy, vz)
- 列3: 语义ID (semantic_id)

**问题**: 体素索引范围[310~697]不是从0开始

---

### 4. 训练脚本（当前未使用方案A）

**文件**: `train_voxelwise_detection.py` (439行)

**标签生成**: 第228行（当前使用随机标签）

```python
# 第228行：当前使用的方法（随机标签）
labels = create_improved_labels_from_voxels(batch, N, device)
```

**需要替换为**:

```python
# 使用精确映射（方案A）
from precise_voxel_mapping_v2 import create_precise_point_labels_from_voxels

labels = create_precise_point_labels_from_voxels(
    points_batch=batch['points'],
    voxel_data_list=[...],  # 从batch['voxel']获取
    anomaly_labels_batch=batch['anomaly_label'],
    cfg=cfg,
    device=device
)
```

---

## 🔧 可能的修复方案

### 方案1：使用EV_POSITION作为锚点

**修改文件**: `precise_voxel_mapping_v2.py`

**修改位置**: 第48-50行

```python
# 当前代码（错误）
grid_origin_x = -((grid_size[0] / 2) + offset_forward) * resolution
grid_origin_y = -(grid_size[1] / 2) * resolution
grid_origin_z = -2.0

# 修改为（尝试方案）
ev_position = cfg.VOXEL.EV_POSITION  # [32, 96, 12]
# 假设车辆在(0,0,0)，EV_POSITION是车辆在体素网格中的位置
grid_origin_x = 0 - ev_position[0] * resolution
grid_origin_y = 0 - ev_position[1] * resolution
grid_origin_z = 0 - ev_position[2] * resolution
```

### 方案2：从实际数据反推

**创建新脚本**: `find_grid_origin.py`

```python
import numpy as np
from muvo.dataset.anovox_dataset import AnoVoxDataset

dataset = AnoVoxDataset(...)
sample = dataset[0]

points = sample['points'].numpy()
voxel = sample['voxel'].numpy()

# 尝试不同的origin，看哪个能让点云和体素对应上
# 思路：找一个已知语义ID的物体（如车辆），
#      在点云和体素中都找到它，然后计算offset
```

### 方案3：查看MUVO原始代码

**思路**: MUVO项目本身应该有体素化代码

**查找位置**: 
```bash
cd /root/autodl-tmp/MUVO/MUVO
grep -r "voxel" --include="*.py" | grep -i "origin\|offset"
grep -r "VOXEL.SIZE" --include="*.py"
```

可能在：
- `muvo/models/` 下的模型实现
- `muvo/utils/` 下的工具函数
- 训练脚本中

---

## 🧪 验证脚本

### 测试当前映射

**文件**: `test_precise_mapping_v2.py` (155行)

**运行**:
```bash
cd /root/autodl-tmp/MUVO/MUVO
python test_precise_mapping_v2.py
```

**当前输出**: 
- 异常点: 0 (0.00%) ❌
- 说明映射失败

**期望输出**:
- 异常点: >0% ✅

### 调试坐标映射

**快速调试命令**:
```bash
cd /root/autodl-tmp/MUVO/MUVO

python -c "
import numpy as np
from muvo.dataset.anovox_dataset import AnoVoxDataset

dataset = AnoVoxDataset(
    data_root='/root/autodl-tmp/datasets/AnoVox/AnoVox_Dynamic_Mono_Town07',
    split='train',
    load_voxel=True
)

sample = dataset[0]
points = sample['points'].numpy()
voxel = sample['voxel'].numpy()

print('点云范围:')
print(f'  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]')
print(f'  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]')
print(f'  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]')

print('\n体素索引范围:')
print(f'  VX: [{voxel[:, 0].min()}, {voxel[:, 0].max()}]')
print(f'  VY: [{voxel[:, 1].min()}, {voxel[:, 1].max()}]')
print(f'  VZ: [{voxel[:, 2].min()}, {voxel[:, 2].max()}]')
"
```

---

## 📚 需要查找的文档

### AnoVox数据集

1. **论文**: 搜索 "AnoVox dataset paper"
2. **GitHub**: 搜索 "AnoVox github"
3. **技术文档**: 查看数据集下载页面

### MUVO项目

1. **原始论文**: MUVO的论文中可能有体素化说明
2. **GitHub**: https://github.com/wayveai/mile (检查是否有体素相关代码)

---

## ✅ 已确认正确的部分

1. ✅ **体素数据加载正常** (`anovox_dataset.py`)
2. ✅ **体素数据格式正确** ([N, 4])
3. ✅ **语义ID存在** (ID 10=车辆, 14,15,18等)
4. ✅ **分辨率0.2米正确**
5. ✅ **坐标变换公式正确** (`floor((coord - origin) / resolution)`)

## ❌ 需要修复的部分

1. ❌ **grid_origin值不正确** (第48-50行)
2. ❓ **不清楚config参数含义** (SIZE, EV_POSITION, OFFSET_FORWARD)
3. ❓ **不清楚体素网格的全局定义**

---

## 🎯 讨论后的行动路径

### 如果导师有AnoVox文档/代码
→ 直接查看正确的坐标系定义  
→ 修改 `precise_voxel_mapping_v2.py` 第48-50行  
→ 重新测试  
→ 集成到训练脚本

### 如果没有文档
→ 尝试方案1（使用EV_POSITION）  
→ 尝试方案2（反推offset）  
→ 如果仍失败，切换到**方案B**（2小时快速方案）

---

**准备完毕，等待您和导师讨论的结果！**

