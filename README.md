# DINOSAUR 3D 点云扩展（含 SPFormer 管线）

> 非官方 PyTorch 实现，基于 DINOSAUR + Invariant Slot Attention，并完成 3D 点云与 SPFormer 超点特征的端到端训练改造。

---

## 1. 项目概述

- **目标**：将 DINOSAUR 的 ISA 模块从 2D 图像 token 迁移到 3D 点云/超点，并提供完整训练脚本。
- **核心能力**：
  - 3D 坐标系适配（ISA & DINOSAURpp 全面升级）。
  - SPFormer → DINOSAUR 训练流水线。
  - 可视化、损失、数据加载与配置工具完备。
  - 所有原独立文档已并入本 README，查阅更集中。

---

## 2. 功能与文件映射

| 模块 | 文件 | 功能 |
|------|------|------|
| 数据加载 | `data/s3dis_dataset.py` | S3DIS 读取、512 超点、增强、collate |
| 模型封装 | `models/wrapper.py` | SPFormer 抽取 + 特征投影 + DINOSAUR |
| 核心模型 | `models/model.py` | ISA / SA / Decoder（含 3D 相对坐标） |
| 损失函数 | `models/losses.py` | 重建 / 熵 / 多样性 / 稀疏性 |
| 可视化 | `utils/visualizer.py` | Slot 分配、重建误差、统计图 |
| 训练脚本 | `train_3d_spformer.py` | 训练/验证/Checkpoint/TensorBoard/可视化 |
| 配置文件 | `config/config_train_spformer.yaml` | 所有超参数 + GPU 绑定 |
| 测试脚本 | `test_3d_isa.py`、`test_train_flow.py` | ISA 单测 & 训练流程模拟 |

---

## 3. 3D ISA 适配摘要

1. **输入输出**：接受 `(B, N, D)` 点特征与 `(B, N, 3)` 归一化坐标，`abs_grid` 动态构建；输出 `(B, N, 768)` 重建、`(B, S, slot_dim)` slots、`(B, S, N)` masks。
2. **3D 相对坐标**：`S_p`、`S_s`、`h`、`g` 全改为 3 维；`slot_encoder.get_rel_grid()` 基于真实坐标计算。
3. **兼容性**：SA 模式无需坐标，保持原逻辑；Decoder、Loss、Wrapper 仅做最小改动。
4. **坐标预处理**：必须将点坐标归一化到 `[-1, 1]`（推荐基于 batch 边界框）。

---

## 4. 代码结构速览

```
Model_Code/src/DINOSAUR
├── config/
├── data/
├── models/
├── utils/
├── train_3d_spformer.py
├── test_3d_isa.py
├── test_train_flow.py
└── README.md  (本文件)
```

---

## 5. 训练准备 Checklist

1. **环境**：`conda activate CloudPoints`
2. **配置数据路径**：
   ```yaml
   data:
     s3dis_root: /abs/path/to/S3DIS/Stanford3dDataset_v1.2_Aligned_Version
     train_areas: [1, 2, 3, 4, 6]
     val_areas: [5]
   ```
3. **SPFormer**：确认 `spformer_config`、`spformer_checkpoint` 可用。
4. **GPU 选择**：`gpu_ids` 控制物理卡，例如 `[0]` / `[0,1]`；脚本会自动设置 `CUDA_VISIBLE_DEVICES`。
5. **批次配置**：`train.batch_size_per_gpu=8` 默认；可配合 `accumulation_steps` 放大有效 batch。

---

## 6. 运行方式

| 场景 | 命令 | 说明 |
|------|------|------|
| 流程自检 | `python train_3d_spformer.py --test_run` | 仅 2 个 epoch，验证数据/模型/可视化 |
| 标准训练 | `python train_3d_spformer.py` | 120 epoch，自动保存 checkpoint & 可视化 |
| 多卡训练 | `torchrun --nproc_per_node=2 train_3d_spformer.py` | 结合 `gpu_ids` 决定具体卡 |

> 建议先执行 `test_3d_isa.py`，确保 ISA 前向、梯度、slot 属性与可视化均正常。

---

## 7. 输出目录

```
checkpoints_spformer/
├── config.yaml
├── epoch_XXX.pth
├── best_model.pth
├── logs/                      # TensorBoard
└── visualizations/
    └── epoch_XXX/
        ├── sample_0_slot_assignment.png
        ├── sample_0_recon_error.png
        └── sample_0_slot_stats.png
```

运行 `tensorboard --logdir checkpoints_spformer/logs` 即可监控。

---

## 8. 调试与监控建议

1. **先跑 `--test_run`**，确认训练循环、可视化、checkpoint 正常输出。  
2. **关注首个 epoch 图像**，检查 slot 是否覆盖不同区域。  
3. **Loss 走势**：`reconstruction` 应明显下降；`slot_diversity` 防止 slot 坍缩。  
4. **Slot 使用率**：若长期空槽，可增大 `slot_dim`、`slot_att_iter` 或调整学习率。  
5. **依赖提示**：`gorilla`, `spconv`, `pointgroup_ops`, `torch_scatter` 需提前编译。

---

## 9. 注意事项

- **坐标归一化**：ISA 依赖 `[-1, 1]`，务必在数据加载或测试脚本中完成。  
- **点数一致**：同一 batch 内的 `num_points` 要一致；数据集加载器已通过采样保证。  
- **SPFormer 冻结策略**：`freeze_spformer=True` 仅训练 projector + DINOSAUR，可在 `unfreeze_after_epoch` 后解冻微调。  
- **GPU 日志**：脚本启动会打印 `[Info] 使用GPU: ...`，确认是否选中期望的显卡。

---

## 10. 测试脚本

### `test_3d_isa.py`
- 生成玩具点云或调用 SPFormer，验证形状、梯度、slot 属性，并输出可视化。
- 关键配置：`TestConfig.use_spformer`, `num_slots`, `num_points`, `visualize`.

### `test_train_flow.py`
- 以模拟数据跑完整 SPFormer→DINOSAUR→损失流程，用于检查训练封装是否工作正常。

---

## 11. 下一步计划

1. 在真实数据集上完成长时间训练并评估精度。  
2. 接入 PointBERT / Point-MAE 等其他 3D 编码器。  
3. 扩展点云语义/实例分割评估脚本。  
4. 增强可视化（交互式点云、Web 视图）。

---

## 12. 参考文献

1. Seitzer, M., et al. *Bridging the Gap to Real-World Object-Centric Learning*, ICLR 2023.  
2. Biza, O., et al. *Invariant Slot Attention: Object Discovery with Slot-Centric Reference Frames*, ICML 2023.  
3. Locatello, F., et al. *Object-Centric Learning with Slot Attention*, NeurIPS 2020.  
4. Oquab, M., et al. *DINOv2: Learning Robust Visual Features without Supervision*, 2023.

---

> 以上内容即原 `TRAINING_READY.md`、`3D适配.md`、`DINOSAUR_代码库结构说明.md` 的整合版本，后续更新请统一维护本 README。