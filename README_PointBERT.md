# PointBERT + DINOSAUR 训练指南

## 概述

本实现使用PointBERT作为冻结的Point Encoder，替代SPFormer进行DINOSAUR训练。

### 主要优势
- ✅ **更简洁的数据流**：不需要预计算超点标签
- ✅ **更强的特征**：ULIP-2多模态预训练（Objaverse+ShapeNet）
- ✅ **真正的批处理**：PointBERT支持批量处理
- ✅ **超点数量完美匹配**：PointBERT默认512组 = DINOSAUR需要的512超点

### 与SPFormer版本对比

| 维度 | SPFormer版 | PointBERT版 |
|------|-----------|-------------|
| **特征维度** | 32 | 384 |
| **数据预处理** | 复杂（聚类+KNN图） | 简单（固定采样） |
| **输入格式** | `(xyz, rgb, sp_labels, sp_coords)` | `(xyzrgb,)` |
| **批处理** | 伪批处理（逐样本循环） | 真批处理 |
| **预训练数据** | ScanNet | Objaverse+ShapeNet |
| **模型大小** | 较小 | 较大（ViT-G） |

---

## 文件结构

```
Model_Code/src/DINOSAUR/
├── config/
│   └── config_train_pointbert.yaml       # 训练配置
├── data/
│   └── s3dis_dataset_pointbert.py        # 简化版数据集
├── models/
│   ├── pointbert_wrapper.py              # PointBERT+DINOSAUR封装
│   ├── model.py                          # DINOSAUR (复用)
│   └── losses.py                         # 损失函数 (复用)
├── utils/
│   └── visualizer.py                     # 可视化工具 (复用)
├── train_3d_pointbert.py                 # 训练脚本
├── test_pointbert_flow.py                # 测试脚本
└── README_PointBERT.md                   # 本文档
```

---

## 快速开始

### 1. 环境准备

确保已安装PointBERT相关依赖：
```bash
cd /home/pbw/data1/3D_PointCloud_Segmentation/PLSG_Net/Model_Code/src/PointBERT
conda activate CloudPoints
```

### 2. 测试数据加载

```bash
cd /home/pbw/data1/3D_PointCloud_Segmentation/PLSG_Net/Model_Code/src/DINOSAUR

# 测试数据集
python data/s3dis_dataset_pointbert.py
```

### 3. 测试完整流程

```bash
# 测试前向传播、损失计算、反向传播
python test_pointbert_flow.py
```

### 4. 开始训练

#### 单卡训练（测试）
```bash
python train_3d_pointbert.py --test_run
```

#### 单卡完整训练
```bash
python train_3d_pointbert.py
```

#### 多卡训练
```bash
# 使用GPU 0
CUDA_VISIBLE_DEVICES=0 python train_3d_pointbert.py

# 或修改配置文件中的gpu_ids
```

---

## 配置说明

### 关键参数 (`config_train_pointbert.yaml`)

```yaml
model:
  pointbert_dim: 384              # PointBERT输出维度
  pointbert_num_groups: 512       # 超点数量
  num_slots: 16                   # Slot数量
  
train:
  batch_size_per_gpu: 6           # batch size（比SPFormer小）
  warmup_epochs: 10               # warmup轮数（比SPFormer多）
  grad_clip_norm: 0.5             # 梯度裁剪
  
data:
  pointbert_checkpoint: .../ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt
  target_points: 8192             # PointBERT输入点数
```

### 超参数调整建议

如果遇到以下情况，可以调整：

#### **训练不稳定（Loss震荡）**
```yaml
train:
  optimizer:
    lr: 0.00005                   # 降低学习率
  warmup_epochs: 15               # 增加warmup
  grad_clip_norm: 0.3             # 更严格的梯度裁剪
```

#### **显存不足**
```yaml
train:
  batch_size_per_gpu: 4           # 降低batch size
  use_amp: True                   # 确保开启混合精度
```

#### **Slot坍塌（所有点分配到少数slot）**
```yaml
loss:
  weights:
    mask_entropy: 0.6             # 增加熵正则
    slot_diversity: 0.6           # 增加多样性损失
```

---

## 输出文件

训练过程会生成以下文件：

```
checkpoints_pointbert/
├── logs/                         # TensorBoard日志
│   └── events.out.tfevents.*
├── visualizations/               # 可视化结果
│   ├── epoch_004/
│   │   ├── sample_0_slot_assignment.png
│   │   ├── sample_0_recon_error.png
│   │   └── sample_0_slot_stats.png
│   └── ...
├── epoch_005.pth                 # 定期checkpoint
├── best_model.pth                # 最佳模型
└── config.yaml                   # 训练配置备份
```

### TensorBoard监控

```bash
tensorboard --logdir checkpoints_pointbert/logs --port 6007
```

然后访问: `http://localhost:6007`

---

## 常见问题

### Q1: 为什么batch size比SPFormer小？
**A**: PointBERT使用ViT-G架构，参数量更大，显存占用更高。建议：
- 单张4090: batch_size=4-6
- 双卡4090: batch_size=6-8 (每卡)

### Q2: 训练速度慢？
**A**: PointBERT前向传播比SPFormer慢2-3倍，这是正常的。优化建议：
- 确保使用混合精度训练 (`use_amp: True`)
- 适当降低可视化频率 (`vis_interval`)
- 使用多GPU训练

### Q3: 出现NaN错误？
**A**: 已添加多层保护，如果仍然出现：
1. 检查 `grad_clip_norm` 是否过大（建议0.3-0.5）
2. 降低学习率（0.00005）
3. 增加warmup轮数（15-20）

### Q4: 如何与SPFormer版本对比？
**A**: 训练完成后，对比以下指标：
- 验证集重建损失 (越低越好)
- Slot assignment可视化 (是否合理分割场景)
- Slot使用率 (避免空slot或少数slot dominant)

---

## 数据流示意图

```
S3DIS房间点云 (N个点)
    ↓ 重采样到8192点
xyzrgb (8192, 6)
    ↓ PointBERT (冻结)
superpoint_features (512, 384)
    ↓ Projector (可训练)
sp_feats_proj (512, 768)
    ↓ DINOSAUR ISA (可训练)
slots (16, 256) + masks (16, 512)
    ↓ Decoder (可训练)
reconstruction (512, 768)
    ↓ Loss计算
MSE + Entropy + Diversity + Sparsity
```

---

## 下一步

训练完成后，可以：
1. 对比PointBERT vs SPFormer的slot assignment质量
2. 分析不同预训练数据的影响
3. 在下游任务中使用训练好的slot表示
4. 尝试解冻PointBERT进行微调（可选）

---

## 参考

- **PointBERT**: Yu et al., "Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling", CVPR 2022
- **ULIP-2**: Xue et al., "ULIP: Learning a Unified Representation of Language, Images, and Point Clouds for 3D Understanding", CVPR 2023
- **DINOSAUR**: Seitzer et al., "Bridging the Gap to Real-World Object-Centric Learning", ICLR 2023
- **ISA**: Biza et al., "Invariant Slot Attention: Object Discovery with Slot-Centric Reference Frames", ICML 2023


