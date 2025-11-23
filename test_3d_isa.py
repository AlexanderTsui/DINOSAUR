"""
3D ISA模块测试和可视化脚本

用途：
1. 验证从2D到3D的修改是否正确（无预训练权重情况下）
2. 可视化slot在3D空间中的分布
3. 检查模型的前向传播、梯度流动和维度匹配

使用方法：
    直接运行此脚本：python test_3d_isa.py
    参数可在 main 函数中直接修改
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# 添加模型路径
sys.path.append(os.path.dirname(__file__))

try:
    from models.model import DINOSAURpp
except ImportError:
    # 如果直接在src/DINOSAUR下运行
    sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
    from model import DINOSAURpp


class TestConfig:
    """测试配置参数"""
    def __init__(self):
        # === Slot Attention 参数 ===
        self.num_slots = 7           # Slot数量
        self.slot_dim = 256          # Slot特征维度
        self.slot_att_iter = 3       # 迭代次数
        self.query_opt = True        # 是否优化Query
        self.ISA = True              # 是否使用ISA（3D位置编码）
        
        # === 点云数据参数 ===
        self.num_points = 1024       # 点云点数
        self.point_feature_dim = 768 # 输入特征维度 (必须与模型硬编码的768匹配)
        self.batch_size = 2          # 测试Batch大小
        self.num_objects = 2         # 生成数据时的模拟物体数
        
        # === 其他参数 ===
        self.token_num = self.num_points  # 3D版本：token_num = num_points
        
        # === 输出配置 ===
        self.visualize = True        # 是否生成可视化图表
        self.output_dir = './visualization/test_results_3d' # 结果保存路径


def generate_toy_point_cloud(config):
    """
    生成玩具点云数据用于测试
    模拟几个高斯分布的簇代表不同的物体
    """
    print(f"\n{'='*60}")
    print("生成玩具点云数据...")
    print(f"{'='*60}")
    
    points_list = []
    features_list = []
    labels_list = []
    
    for b in range(config.batch_size):
        batch_points = []
        batch_features = []
        batch_labels = []
        
        points_per_object = config.num_points // config.num_objects
        
        for obj_id in range(config.num_objects):
            # 每个物体是一个3D高斯分布的点云
            center = np.random.randn(3) * 2  # 随机中心位置
            scale = np.random.rand() * 0.5 + 0.3  # 随机尺度
            
            # 生成坐标
            obj_points = np.random.randn(points_per_object, 3) * scale + center
            
            # 生成特征 (随机初始化，模拟DINO/ViT输出)
            obj_features = np.random.randn(points_per_object, config.point_feature_dim)
            
            obj_labels = np.ones(points_per_object) * obj_id
            
            batch_points.append(obj_points)
            batch_features.append(obj_features)
            batch_labels.append(obj_labels)
        
        # 补齐剩余点数
        current_count = points_per_object * config.num_objects
        if current_count < config.num_points:
            diff = config.num_points - current_count
            batch_points.append(np.random.randn(diff, 3))
            batch_features.append(np.random.randn(diff, config.point_feature_dim))
            batch_labels.append(np.zeros(diff) - 1) # 噪声
        
        # 组合所有物体
        batch_points = np.concatenate(batch_points, axis=0)
        batch_features = np.concatenate(batch_features, axis=0)
        batch_labels = np.concatenate(batch_labels, axis=0)
        
        # 随机打乱顺序（模拟真实点云是无序的）
        indices = np.random.permutation(config.num_points)
        batch_points = batch_points[indices]
        batch_features = batch_features[indices]
        batch_labels = batch_labels[indices]
        
        points_list.append(batch_points)
        features_list.append(batch_features)
        labels_list.append(batch_labels)
    
    points = torch.FloatTensor(np.stack(points_list, axis=0))
    features = torch.FloatTensor(np.stack(features_list, axis=0))
    labels = torch.LongTensor(np.stack(labels_list, axis=0))
    
    print(f"✓ 点云形状: {points.shape}")
    print(f"✓ 特征形状: {features.shape}")
    print(f"✓ 标签形状: {labels.shape}")
    print(f"✓ 坐标范围: [{points.min():.2f}, {points.max():.2f}]")
    
    return points, features, labels


def normalize_point_coords(points):
    """
    将点云坐标归一化到[-1, 1]范围
    这是ISA模块所必需的预处理步骤
    """
    print(f"\n{'='*60}")
    print("归一化点云坐标...")
    print(f"{'='*60}")
    
    # 基于Batch内所有点的边界框归一化
    batch_min = points.min(dim=1, keepdim=True)[0]  # (B, 1, 3)
    batch_max = points.max(dim=1, keepdim=True)[0]  # (B, 1, 3)
    
    # 归一化到 [0, 1]
    normalized = (points - batch_min) / (batch_max - batch_min + 1e-8)
    # 映射到 [-1, 1]
    normalized = normalized * 2 - 1
    
    print(f"✓ 归一化完成，范围: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    return normalized


def test_model_forward(model, points, features, config):
    """
    测试流程核心：模型前向传播与验证
    """
    print(f"\n{'='*60}")
    print("开始模型测试流程")
    print(f"{'='*60}")
    
    # 1. 形状验证
    print("\n[步骤1] 形状验证...")
    try:
        with torch.no_grad():
            reconstruction, slots, masks = model(features, points)
            
        print(f"  输入特征: {features.shape}")
        print(f"  输入坐标: {points.shape}")
        print(f"  输出重建: {reconstruction.shape}")
        print(f"  输出Slots: {slots.shape}")
        print(f"  输出Masks: {masks.shape}")
        
        assert reconstruction.shape == (config.batch_size, config.num_points, 768), "重建形状错误"
        assert slots.shape == (config.batch_size, config.num_slots, config.slot_dim), "Slots形状错误"
        assert masks.shape == (config.batch_size, config.num_slots, config.num_points), "Masks形状错误"
        print("  ✅ 形状验证通过")
        
    except Exception as e:
        print(f"  ❌ 形状验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None

    # 2. 梯度流动验证
    print("\n[步骤2] 梯度流动验证...")
    try:
        features_grad = features.clone().requires_grad_(True)
        # 注意：points通常不需要梯度，因为它是输入坐标
        
        model.zero_grad()
        reconstruction, slots, masks = model(features_grad, points)
        
        # 构建一个简单的损失函数
        loss = reconstruction.sum() + slots.sum()
        loss.backward()
        
        has_grad = features_grad.grad is not None
        grad_norm = features_grad.grad.norm().item() if has_grad else 0
        
        print(f"  特征梯度存在: {has_grad}")
        print(f"  梯度范数: {grad_norm:.4f}")
        
        assert has_grad, "梯度未反向传播到输入特征"
        assert grad_norm > 0, "梯度为零，可能存在断开的计算图"
        print("  ✅ 梯度验证通过")
        
    except Exception as e:
        print(f"  ❌ 梯度验证失败: {e}")
        return False, None, None, None

    # 3. Slot属性检查 (仅在ISA模式下)
    print("\n[步骤3] Slot属性检查...")
    if config.ISA:
        try:
            with torch.no_grad():
                # 获取内部attention和slot参数
                slots_enc, attn = model.slot_encoder(features, points)
                
                # 手动计算Slot中心 (S_p) 用于验证
                attn_expanded = attn.unsqueeze(2)  # (B, S, 1, N)
                abs_grid = points.unsqueeze(1).expand(config.batch_size, config.num_slots, config.num_points, 3)
                
                # 加权平均位置
                S_p = torch.einsum('bsjd,bsij->bsd', abs_grid, attn_expanded)
                
                # 检查多样性
                diversity = S_p.std(dim=1).mean().item()
                print(f"  Slot空间分布多样性(Std): {diversity:.4f}")
                
                if diversity < 0.01:
                    print("  ⚠️  警告: Slot中心聚集在一起，可能是初始化问题（但在无训练权重下属正常现象）")
                else:
                    print("  ✅ Slot分布具有一定的空间差异")
                    
        except Exception as e:
            print(f"  ❌ Slot属性检查失败: {e}")
            pass # 不中断流程

    return True, reconstruction, slots, masks


def visualize_results(points, masks, slots, save_dir):
    """
    生成可视化结果
    """
    print(f"\n{'='*60}")
    print("生成可视化报告")
    print(f"{'='*60}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 取第一个样本
    points_np = points[0].cpu().numpy()
    masks_np = masks[0].detach().cpu().numpy() # (S, N)
    
    num_slots = masks_np.shape[0]
    
    # 生成每个Slot的专属颜色
    if num_slots <= 10:
        cmap = plt.get_cmap('tab10')
        colors_lookup = np.array([cmap(i) for i in range(num_slots)])
    elif num_slots <= 20:
        cmap = plt.get_cmap('tab20')
        colors_lookup = np.array([cmap(i) for i in range(num_slots)])
    else:
        # 如果slot太多，使用hsv均匀分布
        cmap = plt.get_cmap('hsv')
        colors_lookup = np.array([cmap(i / num_slots) for i in range(num_slots)])
    
    # 1. Slot Assignment 可视化
    fig = plt.figure(figsize=(15, 6))
    
    # 左图：点云
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], s=5, c='gray', alpha=0.5)
    ax1.set_title('Input Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 右图：Slot分配 (Argmax)
    ax2 = fig.add_subplot(122, projection='3d')
    slot_ids = masks_np.argmax(axis=0) # (N,)
    
    # 根据Slot ID映射颜色
    point_colors = colors_lookup[slot_ids]
    
    scatter = ax2.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], s=10, c=point_colors)
    ax2.set_title(f'Slot Assignment (Total {num_slots} Slots)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # 添加Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Slot {i}',
                          markerfacecolor=colors_lookup[i], markersize=8)
                   for i in range(num_slots)]
    
    # 将Legend放在图外
    ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1.0), title="Slots")
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'visualization.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"✓ 可视化已保存: {save_path}")


def main():
    # 1. 配置
    config = TestConfig()
    
    # 用户可以在这里修改参数
    # config.batch_size = 4
    # config.num_points = 2048
    
    print(f"测试配置:")
    print(f"  - ISA模式: {config.ISA}")
    print(f"  - Slot数量: {config.num_slots}")
    print(f"  - 特征维度: {config.point_feature_dim}")
    
    # 2. 准备数据
    points, features, labels = generate_toy_point_cloud(config)
    points_norm = normalize_point_coords(points)
    
    # 3. 初始化模型
    print(f"\n{'='*60}")
    print("初始化模型...")
    print(f"{'='*60}")
    try:
        model = DINOSAURpp(config)
        model.eval() # 默认为eval模式，但测试梯度时需要注意
        print("✓ 模型初始化成功")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return

    # 4. 运行测试
    success, recon, slots, masks = test_model_forward(model, points_norm, features, config)
    
    if success:
        print(f"\n{'='*60}")
        print("🎉 模块测试通过！逻辑通路正常。")
        print(f"{'='*60}")
        
        if config.visualize:
            visualize_results(points_norm, masks, slots, config.output_dir)
    else:
        print(f"\n{'='*60}")
        print("⚠️ 测试发现问题，请检查日志。")
        print(f"{'='*60}")

if __name__ == '__main__':
    main()
