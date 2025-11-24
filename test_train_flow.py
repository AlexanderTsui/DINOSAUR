"""
训练流程测试脚本（使用模拟数据）
验证代码逻辑是否正确
"""

import os
import sys
import torch
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '../SPFormer'))

from models.model import DINOSAURpp
from models.wrapper import FeatureProjector, SPFormerDINOSAUR
from models.losses import DINOSAURLoss
from test_3d_isa_with_spformer import TestSPFormerExtractor

print("\n" + "="*60)
print("训练流程测试 (模拟数据)")
print("="*60)

# 1. 创建模型
print("\n[1] 创建模型...")

class Args:
    def __init__(self):
        self.num_slots = 16
        self.slot_dim = 256
        self.slot_att_iter = 3
        self.query_opt = True
        self.ISA = True
        self.token_num = 512
        self.num_points = 512
        self.point_feature_dim = 384

args = Args()

# SPFormer
spformer_config = os.path.join(current_dir, '../SPFormer/configs/spf_scannet.yaml')
spformer_extractor = TestSPFormerExtractor(spformer_config, device='cuda')

# Projector (32维 → 768维)
projector = FeatureProjector(in_dim=32, out_dim=768)

# DINOSAUR
dinosaur = DINOSAURpp(args)

# 封装
model = SPFormerDINOSAUR(spformer_extractor, projector, dinosaur).cuda()

print("✓ 模型创建成功")

# 2. 创建损失函数
print("\n[2] 创建损失函数...")

loss_weights = {
    'reconstruction': 1.0,
    'mask_entropy': 0.15,
    'slot_diversity': 0.08,
    'mask_sparsity': 0.05
}

criterion = DINOSAURLoss(loss_weights)
print("✓ 损失函数创建成功")

# 3. 生成模拟数据
print("\n[3] 生成模拟数据...")

batch_size = 2
n_points = 2000
n_superpoints = 512

# 模拟batch数据
xyz_full_list = []
rgb_full_list = []
sp_labels_list = []
sp_coords_list = []

for b in range(batch_size):
    # 点云
    xyz = torch.randn(n_points, 3).cuda()
    rgb = torch.rand(n_points, 3).cuda() * 2 - 1
    
    # 超点标签
    sp_labels = torch.randint(0, n_superpoints, (n_points,)).cuda()
    
    # 超点中心
    sp_coords = torch.randn(n_superpoints, 3).cuda()
    sp_coords = (sp_coords - sp_coords.min(0)[0]) / (sp_coords.max(0)[0] - sp_coords.min(0)[0] + 1e-8)
    sp_coords = sp_coords * 2 - 1
    
    xyz_full_list.append(xyz)
    rgb_full_list.append(rgb)
    sp_labels_list.append(sp_labels)
    sp_coords_list.append(sp_coords)

sp_coords_batch = torch.stack(sp_coords_list)

print(f"✓ 生成了 {batch_size} 个样本")
print(f"  - 点云: ({n_points}, 3)")
print(f"  - 超点坐标: ({n_superpoints}, 3)")

# 4. 前向传播
print("\n[4] 测试前向传播...")

model.eval()
with torch.no_grad():
    reconstruction, slots, masks, sp_feats_proj = model(
        xyz_full_list,
        rgb_full_list,
        sp_labels_list,
        sp_coords_batch
    )

print(f"✓ 前向传播成功")
print(f"  - reconstruction: {reconstruction.shape}")
print(f"  - slots: {slots.shape}")
print(f"  - masks: {masks.shape}")
print(f"  - sp_feats_proj: {sp_feats_proj.shape}")

# 5. 计算损失
print("\n[5] 测试损失计算...")

model.train()
reconstruction_train, slots_train, masks_train, sp_feats_proj_train = model(
    xyz_full_list,
    rgb_full_list,
    sp_labels_list,
    sp_coords_batch
)

loss, loss_dict = criterion(reconstruction_train, sp_feats_proj_train, slots_train, masks_train)

print(f"✓ 损失计算成功")
print(f"  - total_loss: {loss_dict['total']:.6f}")
print(f"  - reconstruction: {loss_dict['reconstruction']:.6f}")
print(f"  - mask_entropy: {loss_dict['mask_entropy']:.6f}")
print(f"  - slot_diversity: {loss_dict['slot_diversity']:.6f}")
print(f"  - mask_sparsity: {loss_dict['mask_sparsity']:.6f}")

# 6. 测试反向传播
print("\n[6] 测试反向传播...")

optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=2e-4)
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), 1.0)
optimizer.step()

print(f"✓ 反向传播成功")

# 7. 测试可视化（可选）
print("\n[7] 测试可视化...")

try:
    sys.path.insert(0, os.path.join(current_dir, 'utils'))
    from visualizer import visualize_slot_assignment
    
    xyz_np = xyz_full_list[0].cpu().numpy()
    sp_labels_np = sp_labels_list[0].cpu().numpy()
    masks_np = masks[0].detach().cpu().numpy()
    
    visualize_slot_assignment(
        xyz_np, sp_labels_np, masks_np,
        'test_visualization.png',
        num_slots=16
    )
    print("✓ 可视化测试成功: test_visualization.png")
except Exception as e:
    print(f"⚠️  可视化测试跳过: {e}")

# 总结
print("\n" + "="*60)
print("🎉 所有测试通过！训练流程代码逻辑正确")
print("="*60)
print("\n提示:")
print("1. 请确保S3DIS数据集路径正确")
print("2. 数据集路径在config文件中修改: data.s3dis_root")
print("3. 完整训练命令:")
print("   python train_3d_spformer.py")
print("4. 测试运行(2 epochs):")
print("   python train_3d_spformer.py --test_run")
print()

