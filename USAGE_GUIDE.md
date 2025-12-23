# 对比学习损失集成与特征分析工具

## 📝 概述

本次更新完成了以下工作：

### 1. ✅ 集成对比学习损失到DINOSAUR
- **新增 `ContrastiveLoss` 类** (`DINOSAUR/models/losses.py`)
  - **Intra-slot Compactness**: slot内特征紧致性（基于prototype）
  - **Inter-slot Separation**: 不同slot间的分离性（多样性）
  - **Foreground-Background Contrast**: 前景-背景对比（Two-Stage专用）

### 2. ✅ 修改训练代码支持对比损失
- 自动检测Two-Stage模式
- 在损失计算时传递 `use_two_stage` 参数
- 验证阶段也支持对比损失

### 3. ✅ 更新训练配置文件
- 添加对比损失权重配置
- 添加对比损失的warmup策略
- 添加对比损失相关参数（temperature等）

### 4. ✅ 编写特征分析工具
- **完整的特征提取和可视化流程**
- **诊断报告自动生成**
- **多维度分析：encoder特征、slot绑定、slot特征**

---

## 🚀 快速开始

### 方法1：运行特征分析（推荐先做）

#### 分析单阶段模型
```bash
cd /data1/cbw/3D_PointCloud_Segmentation/PLSG_Net/Model_Code/src/DINOSAUR

./run_feature_analysis.sh \
  --config config/config_train_concerto_scannet.yaml \
  --checkpoint checkpoints/checkpoints_concerto/concerto_scannet_origin/epoch_200.pth \
  --dataset scannet \
  --num_samples 20 \
  --output_dir analysis_results/single_stage_analysis
```

#### 分析Two-Stage模型
```bash
./run_feature_analysis.sh \
  --config config/config_train_concerto_scannet.yaml \
  --checkpoint checkpoints/checkpoints_concerto/concerto_scannet_origin_2stage/best_model.pth \
  --dataset scannet \
  --num_samples 20 \
  --output_dir analysis_results/two_stage_analysis
```

#### 查看分析结果
```bash
# 查看诊断报告
cat analysis_results/single_stage_analysis/00_DIAGNOSIS_REPORT.txt

# 查看所有生成的图片
ls analysis_results/single_stage_analysis/*.png
```

**生成的可视化文件：**
- `01_pca_analysis.png`: PCA特征分析（方差解释+前2个主成分）
- `02_tsne_analysis.png`: t-SNE特征可视化（按类别着色）
- `03_slot_occupancy.png`: Slot占用率分析（Bar+Heatmap）
- `04_bg_fg_separation.png`: 背景/前景分离（仅Two-Stage）
- `06_slot_features_pca.png`: Slot特征PCA
- `07_slot_similarity_distribution.png`: Slot相似度分布

---

### 方法2：使用对比损失训练

#### Step 1: 确认配置文件已更新

检查 `config/config_train_concerto_scannet.yaml` 中是否包含：

```yaml
loss:
  weights:
    feat_rec: 1.0
    compact: 0
    entropy: 0
    min_usage: 0
    diversity: 0.2
    # 新增：对比学习损失
    contrastive_compact: 0.5      # Slot内紧致性
    contrastive_separate: 0.3     # Slot间分离性
    contrastive_fg_bg: 0.2        # 前景-背景对比（Two-Stage）

  warmup:
    items:
      # ... 其他warmup配置 ...
      contrastive_compact:
        enabled: True
        start_epoch: 20
        warmup_epochs: 30
        start_weight: 0.0
      contrastive_separate:
        enabled: True
        start_epoch: 20
        warmup_epochs: 30
        start_weight: 0.0
      contrastive_fg_bg:
        enabled: True
        start_epoch: 30
        warmup_epochs: 30
        start_weight: 0.0

  params:
    # ... 其他参数 ...
    contrastive_temperature: 0.07
```

#### Step 2: 训练Two-Stage模型（推荐）

```bash
cd /data1/cbw/3D_PointCloud_Segmentation/PLSG_Net/Model_Code/src/DINOSAUR

python train_3d_mask3d.py \
  --config config/config_train_concerto_scannet.yaml \
  --gpu_ids 7
```

**训练时会自动：**
- 检测Two-Stage模式
- 启用对比损失（如果权重>0）
- 在TensorBoard中记录对比损失

#### Step 3: 监控训练

```bash
# 查看TensorBoard
tensorboard --logdir checkpoints/checkpoints_concerto/concerto_scannet_origin_2stage/logs

# 检查训练日志
tail -f checkpoints/checkpoints_concerto/concerto_scannet_origin_2stage/logs/train.log
```

---

## 📊 特征分析详解

### 分析器功能

`analyze_features.py` 提供以下分析：

#### 1. Encoder特征质量分析
- **PCA方差解释**: 特征的主要变化方向
- **t-SNE可视化**: 2D空间中的特征聚类
- **Silhouette Score**: 特征可分性指标 ([-1, 1], 越大越好)
- **类内/类间距离**: 分离比 = 类间距离 / 类内距离 (越大越好)

**指标解读:**
- Silhouette Score > 0.3: 特征质量良好
- Silhouette Score 0.1~0.3: 特征质量一般
- Silhouette Score < 0.1: 特征质量较差

- 分离比 > 2.0: 类间远大于类内，特征区分性好
- 分离比 1.0~2.0: 类间/类内接近，需改进
- 分离比 < 1.0: 类内大于类间，特征混乱

#### 2. Slot绑定分析
- **Slot占用率分布**: 每个slot平均关注多少点
- **Slot Overlap**: slots之间的重叠程度（越小越好，< 0.3 为佳）
- **背景/前景分离**: Two-Stage模式下的bg/fg占用率对比

**问题诊断:**
- **不均衡占用**: 某些slots占用率过高（>0.3），某些过低（<0.01）
  → 说明slot collapse，部分slots未被有效利用
- **高Overlap (>0.5)**: 多个slots关注同一区域
  → 说明slots没有学到不同的物体表征
- **背景slot占用低 (<0.3)**: Two-Stage中背景slot未有效捕获背景
  → 导致前景slots被迫绑定背景

#### 3. Slot特征分析
- **Slot特征PCA**: 不同slots的表征是否多样
- **Pairwise相似度**: slots之间的余弦相似度分布
  - 平均相似度 < 0.3: slots学到不同表征（好）
  - 平均相似度 > 0.5: slots高度相似（坏，slot collapse）

---

## 🎯 改进建议优先级

基于分析结果，按优先级采取行动：

### Priority 1: 切换到Two-Stage（立即）⭐⭐⭐⭐⭐
如果当前使用单阶段模型：
```bash
# 1. 检查Two-Stage checkpoint是否可用
ls checkpoints/checkpoints_concerto/concerto_scannet_origin_2stage/

# 2. 如果没有，重新训练（确保配置中 two_stage: true）
python train_3d_mask3d.py --config config/config_train_concerto_scannet.yaml --gpu_ids 7

# 3. 评估Two-Stage效果
./run_feature_analysis.sh --checkpoint .../concerto_scannet_origin_2stage/best_model.pth
```

**预期效果：**
- Slot 0占用率应该在0.5~0.8（背景）
- Slots 1-N占用率更均衡
- Slot Overlap显著降低

### Priority 2: 启用对比损失（短期）⭐⭐⭐⭐
```yaml
# 编辑 config/config_train_concerto_scannet.yaml
loss:
  weights:
    contrastive_compact: 0.5
    contrastive_separate: 0.3
    contrastive_fg_bg: 0.2  # Two-Stage专用
```

**预期效果：**
- Slot特征平均相似度降低（<0.3）
- 分离比提升（>2.0）
- Slot Overlap降低

### Priority 3: 调整聚类策略（中期）⭐⭐⭐
```yaml
# 编辑 Unsupervised_Seg/config.yaml
clustering:
  type: kmeans  # 或 hdbscan

hdbscan:
  use_size_feature: true  # 启用size特征
  size_weight: 0.5
  use_spatial_feature: true  # 启用spatial特征
  spatial_weight: 0.5
  normalize_features: true  # 特征归一化
```

**并且：** 修改聚类代码，如果使用Two-Stage，只对前景slots聚类。

---

## 🔍 问题定位流程

### 如果无监督分割效果不好：

1. **运行特征分析**
   ```bash
   ./run_feature_analysis.sh --checkpoint <你的checkpoint>
   ```

2. **查看诊断报告**
   ```bash
   cat analysis_results/*/00_DIAGNOSIS_REPORT.txt
   ```

3. **根据报告定位问题：**
   - **Silhouette Score < 0.1**
     → Encoder特征质量差，考虑：
     - 换更强的encoder（Concerto > LogoSP > Mask3D）
     - 增加训练epochs
     - 调整projector深度

   - **Slot Overlap > 0.5**
     → Slot collapse，考虑：
     - 启用Two-Stage
     - 增加diversity loss权重
     - 启用对比损失

   - **背景slot占用 < 0.3 (Two-Stage)**
     → 背景/前景分离失败，考虑：
     - 调整 `two_stage_bg_init_scale` (default: 2.0)
     - 增加 `bg_area` loss权重
     - 增加 `contrastive_fg_bg` 权重

4. **应用改进措施并重新训练**

5. **再次运行分析对比效果**

---

## 📁 文件说明

### 新增/修改的文件

#### 核心代码
- `DINOSAUR/models/losses.py`: 新增 `ContrastiveLoss` 类
- `DINOSAUR/train_3d_mask3d.py`: 修改以支持对比损失

#### 配置文件
- `DINOSAUR/config/config_train_concerto_scannet.yaml`: 添加对比损失配置

#### 分析工具
- `DINOSAUR/analyze_features.py`: 特征分析主脚本
- `DINOSAUR/run_feature_analysis.sh`: 便捷运行脚本
- `DINOSAUR/USAGE_GUIDE.md`: 本文档

---

## 💡 常见问题

### Q1: 对比损失会不会影响重建质量？
A: 对比损失是在projected features上计算的，使用 `stop_grad` 策略与compactness loss一致。通过warmup策略（从epoch 20开始），不会影响早期的重建学习。

### Q2: Two-Stage和对比损失哪个更重要？
A: **Two-Stage更优先**。它是结构性的改进，强制前景-背景分离。对比损失是辅助，增强slot表征的多样性。建议两者结合使用。

### Q3: 特征分析需要多长时间？
A: 取决于样本数量：
- 20个样本：~5-10分钟
- 50个样本：~15-20分钟
- 100个样本：~30-40分钟

### Q4: 对比损失的权重如何调整？
A: 默认配置是：
- `contrastive_compact: 0.5`
- `contrastive_separate: 0.3`
- `contrastive_fg_bg: 0.2`

如果发现：
- Slot overlap仍然很高 → 增加 `contrastive_separate` (0.5~0.8)
- 前景-背景混淆严重 → 增加 `contrastive_fg_bg` (0.3~0.5)
- 训练不稳定 → 降低所有权重，延长warmup

---

## 📞 下一步

1. **立即运行特征分析**，对比单阶段vs两阶段模型
2. **查看诊断报告**，定位具体问题
3. **根据建议调整**配置并重新训练
4. **再次分析**，验证改进效果

如有问题，请查看生成的诊断报告或联系开发者。

---

**祝实验顺利！** 🚀
