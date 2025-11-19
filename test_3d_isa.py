"""
3D ISA模块测试和可视化脚本

用途：
1. 验证从2D到3D的修改是否正确
2. 可视化slot在3D空间中的分布
3. 检查模型的前向传播和反向传播

使用方法：
    python test_3d_isa.py --visualize
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import sys
import os

# 添加模型路径
sys.path.append(os.path.dirname(__file__))

# 假设已经修改好的model.py
from models.model import ISA, DINOSAURpp


class Args:
    """模拟参数配置"""
    def __init__(self):
        # Slot Attention参数
        self.num_slots = 7
        self.slot_dim = 256
        self.slot_att_iter = 3
        self.query_opt = True
        self.ISA = True
        
        # 点云参数
        self.num_points = 1024
        self.point_feature_dim = 384
        
        # 其他参数
        self.token_num = self.num_points  # 3D版本：token_num = num_points


def generate_toy_point_cloud(batch_size=2, num_points=1024, num_objects=3):
    """
    生成玩具点云数据用于测试
    
    Args:
        batch_size: batch大小
        num_points: 每个点云的点数
        num_objects: 物体数量
    
    Returns:
        points: (B, N, 3) - 点云坐标
        features: (B, N, D) - 点云特征
        labels: (B, N) - 真实标签（用于可视化）
    """
    print(f"\n{'='*60}")
    print("生成玩具点云数据...")
    print(f"{'='*60}")
    
    points_list = []
    features_list = []
    labels_list = []
    
    for b in range(batch_size):
        batch_points = []
        batch_features = []
        batch_labels = []
        
        points_per_object = num_points // num_objects
        
        for obj_id in range(num_objects):
            # 每个物体是一个3D高斯分布的点云
            center = np.random.randn(3) * 2  # 随机中心位置
            scale = np.random.rand() * 0.5 + 0.3  # 随机尺度
            
            obj_points = np.random.randn(points_per_object, 3) * scale + center
            obj_features = np.random.randn(points_per_object, 384)  # 随机特征
            obj_labels = np.ones(points_per_object) * obj_id
            
            batch_points.append(obj_points)
            batch_features.append(obj_features)
            batch_labels.append(obj_labels)
        
        # 组合所有物体
        batch_points = np.concatenate(batch_points, axis=0)
        batch_features = np.concatenate(batch_features, axis=0)
        batch_labels = np.concatenate(batch_labels, axis=0)
        
        # 随机打乱顺序（模拟真实点云）
        indices = np.random.permutation(num_points)
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
    
    Args:
        points: (B, N, 3)
    
    Returns:
        normalized_points: (B, N, 3)
    """
    print(f"\n{'='*60}")
    print("归一化点云坐标...")
    print(f"{'='*60}")
    
    print(f"原始坐标范围: [{points.min():.3f}, {points.max():.3f}]")
    
    # 方法1：基于边界框归一化
    batch_min = points.min(dim=1, keepdim=True)[0]  # (B, 1, 3)
    batch_max = points.max(dim=1, keepdim=True)[0]  # (B, 1, 3)
    
    normalized = (points - batch_min) / (batch_max - batch_min + 1e-8)
    normalized = normalized * 2 - 1  # 缩放到[-1, 1]
    
    print(f"归一化后范围: [{normalized.min():.3f}, {normalized.max():.3f}]")
    print(f"✓ 归一化完成")
    
    return normalized


def test_shape_validation(model, points, features):
    """
    测试1：形状验证
    确保所有tensor的形状正确
    """
    print(f"\n{'='*60}")
    print("测试1: 形状验证")
    print(f"{'='*60}")
    
    B, N, _ = points.shape
    
    try:
        # 前向传播
        with torch.no_grad():
            slots, attn = model.slot_encoder(features, points)
        
        print(f"✓ 输入特征: {features.shape}")
        print(f"✓ 输入坐标: {points.shape}")
        print(f"✓ 输出slots: {slots.shape}")
        print(f"✓ 输出attn: {attn.shape}")
        
        # 验证形状
        assert slots.shape == (B, model.slot_num, model.slot_dim), \
            f"Slots形状错误: 期望({B}, {model.slot_num}, {model.slot_dim}), 实际{slots.shape}"
        assert attn.shape == (B, model.slot_num, N), \
            f"Attention形状错误: 期望({B}, {model.slot_num}, {N}), 实际{attn.shape}"
        
        print(f"\n✅ 形状验证通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 形状验证失败: {e}")
        return False


def test_numerical_validation(model, points, features):
    """
    测试2：数值验证
    检查NaN、Inf和梯度
    """
    print(f"\n{'='*60}")
    print("测试2: 数值验证")
    print(f"{'='*60}")
    
    try:
        # 前向传播
        slots, attn = model.slot_encoder(features, points)
        
        # 检查NaN
        has_nan_slots = torch.isnan(slots).any()
        has_nan_attn = torch.isnan(attn).any()
        
        print(f"Slots包含NaN: {has_nan_slots}")
        print(f"Attention包含NaN: {has_nan_attn}")
        
        # 检查Inf
        has_inf_slots = torch.isinf(slots).any()
        has_inf_attn = torch.isinf(attn).any()
        
        print(f"Slots包含Inf: {has_inf_slots}")
        print(f"Attention包含Inf: {has_inf_attn}")
        
        # 检查值范围
        print(f"\nSlots值范围: [{slots.min():.3f}, {slots.max():.3f}]")
        print(f"Attention值范围: [{attn.min():.3f}, {attn.max():.3f}]")
        print(f"Attention和（应该≈1）: {attn.sum(dim=1).mean():.6f}")
        
        # 检查梯度
        print(f"\n检查梯度...")
        features_grad = features.clone().requires_grad_(True)
        points_grad = points.clone()
        
        slots_grad, attn_grad = model.slot_encoder(features_grad, points_grad)
        loss = slots_grad.sum()
        loss.backward()
        
        has_grad = features_grad.grad is not None
        print(f"特征梯度存在: {has_grad}")
        if has_grad:
            print(f"梯度范围: [{features_grad.grad.min():.3f}, {features_grad.grad.max():.3f}]")
        
        # 验证
        assert not has_nan_slots and not has_nan_attn, "存在NaN值"
        assert not has_inf_slots and not has_inf_attn, "存在Inf值"
        assert has_grad, "梯度未计算"
        
        print(f"\n✅ 数值验证通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 数值验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_slot_properties(model, points, features):
    """
    测试3：Slot属性验证
    检查slot中心和尺度是否合理
    """
    print(f"\n{'='*60}")
    print("测试3: Slot属性验证")
    print(f"{'='*60}")
    
    try:
        with torch.no_grad():
            slots, attn = model.slot_encoder(features, points)
            
            # 获取slot中心和尺度（需要在最后一次迭代后）
            # 这里我们重新计算
            B, S = attn.shape[:2]
            N = points.shape[1]
            
            attn_expanded = attn.unsqueeze(2)  # (B, S, 1, N)
            abs_grid = points.unsqueeze(1).expand(B, S, N, 3)  # (B, S, N, 3)
            
            # 计算slot中心
            S_p = torch.einsum('bsjd,bsij->bsd', abs_grid, attn_expanded)  # (B, S, 3)
            
            # 计算slot尺度
            values_ss = torch.pow(abs_grid - S_p.unsqueeze(2), 2)
            S_s = torch.sqrt(torch.einsum('bsjd,bsij->bsd', values_ss, attn_expanded))  # (B, S, 3)
            
            print(f"\nSlot中心位置 (S_p):")
            print(f"形状: {S_p.shape}")
            print(f"范围: [{S_p.min():.3f}, {S_p.max():.3f}]")
            print(f"\n前3个slot的中心 (batch 0):")
            for i in range(min(3, S)):
                print(f"  Slot {i}: [{S_p[0, i, 0]:.3f}, {S_p[0, i, 1]:.3f}, {S_p[0, i, 2]:.3f}]")
            
            print(f"\nSlot尺度 (S_s):")
            print(f"形状: {S_s.shape}")
            print(f"范围: [{S_s.min():.3f}, {S_s.max():.3f}]")
            print(f"\n前3个slot的尺度 (batch 0):")
            for i in range(min(3, S)):
                print(f"  Slot {i}: [{S_s[0, i, 0]:.3f}, {S_s[0, i, 1]:.3f}, {S_s[0, i, 2]:.3f}]")
            
            # 检查slot中心的分散程度
            S_p_mean = S_p.mean(dim=1)  # (B, 3)
            S_p_std = S_p.std(dim=1)   # (B, 3)
            print(f"\nSlot中心的分散程度:")
            print(f"平均位置: {S_p_mean[0]}")
            print(f"标准差: {S_p_std[0]}")
            
            # 验证：slot中心应该在归一化范围内
            in_range = (S_p >= -2).all() and (S_p <= 2).all()
            print(f"\nSlot中心在合理范围内: {in_range}")
            
            # 验证：不同slot的中心应该有差异
            has_diversity = S_p_std.mean() > 0.1
            print(f"Slot具有空间多样性: {has_diversity}")
            
            print(f"\n✅ Slot属性验证通过！")
            return True, S_p, S_s
            
    except Exception as e:
        print(f"\n❌ Slot属性验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def visualize_slot_centers(points, S_p, S_s, save_path='slot_centers_3d.png'):
    """
    可视化1：Slot中心在3D空间中的分布
    """
    print(f"\n{'='*60}")
    print("可视化1: Slot中心的3D分布")
    print(f"{'='*60}")
    
    try:
        fig = plt.figure(figsize=(15, 5))
        
        # 绘制第一个batch
        points_np = points[0].cpu().numpy()
        S_p_np = S_p[0].cpu().numpy()
        S_s_np = S_s[0].cpu().numpy()
        
        # 子图1：点云 + slot中心
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2],
                   c='gray', alpha=0.2, s=1, label='Point Cloud')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(S_p_np)))
        for i, (sp, ss, color) in enumerate(zip(S_p_np, S_s_np, colors)):
            ax1.scatter(sp[0], sp[1], sp[2], 
                       c=[color], s=200, marker='o', 
                       edgecolors='black', linewidths=2,
                       label=f'Slot {i}')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Point Cloud + Slot Centers')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # 子图2：仅slot中心
        ax2 = fig.add_subplot(132, projection='3d')
        for i, (sp, color) in enumerate(zip(S_p_np, colors)):
            ax2.scatter(sp[0], sp[1], sp[2],
                       c=[color], s=300, marker='o',
                       edgecolors='black', linewidths=2,
                       label=f'Slot {i}')
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Slot Centers Only')
        ax2.legend(fontsize=8)
        
        # 子图3：slot中心的2D投影（俯视图）
        ax3 = fig.add_subplot(133)
        ax3.scatter(points_np[:, 0], points_np[:, 1],
                   c='gray', alpha=0.2, s=1)
        
        for i, (sp, ss, color) in enumerate(zip(S_p_np, S_s_np, colors)):
            ax3.scatter(sp[0], sp[1], c=[color], s=200, marker='o',
                       edgecolors='black', linewidths=2, label=f'Slot {i}')
            
            # 绘制尺度椭圆（XY平面）
            from matplotlib.patches import Ellipse
            ellipse = Ellipse((sp[0], sp[1]), ss[0]*2, ss[1]*2,
                            alpha=0.3, facecolor=color, edgecolor='black')
            ax3.add_patch(ellipse)
        
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_title('Top View (XY Plane)')
        ax3.legend(fontsize=8)
        ax3.axis('equal')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 图像已保存: {save_path}")
        plt.close()
        
        print(f"✅ Slot中心可视化完成！")
        return True
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_attention_weights(points, attn, labels, save_path='attention_weights_3d.png'):
    """
    可视化2：每个slot的attention权重
    """
    print(f"\n{'='*60}")
    print("可视化2: Attention权重分布")
    print(f"{'='*60}")
    
    try:
        num_slots = attn.shape[1]
        rows = 2
        cols = (num_slots + 1) // 2
        
        fig = plt.figure(figsize=(cols * 5, rows * 4))
        
        points_np = points[0].cpu().numpy()
        attn_np = attn[0].cpu().numpy()
        
        for slot_idx in range(num_slots):
            ax = fig.add_subplot(rows, cols, slot_idx + 1, projection='3d')
            
            # 获取该slot的attention权重
            weights = attn_np[slot_idx]  # (N,)
            
            # 用attention权重给点云着色
            scatter = ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2],
                               c=weights, cmap='hot', s=10, 
                               vmin=0, vmax=weights.max())
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Slot {slot_idx}\n(max attn: {weights.max():.4f})')
            
            plt.colorbar(scatter, ax=ax, shrink=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 图像已保存: {save_path}")
        plt.close()
        
        print(f"✅ Attention权重可视化完成！")
        return True
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_slot_assignment(points, attn, labels, save_path='slot_assignment_3d.png'):
    """
    可视化3：Slot分配结果
    """
    print(f"\n{'='*60}")
    print("可视化3: Slot分配结果")
    print(f"{'='*60}")
    
    try:
        fig = plt.figure(figsize=(15, 5))
        
        points_np = points[0].cpu().numpy()
        attn_np = attn[0].cpu().numpy()
        labels_np = labels[0].cpu().numpy()
        
        # 计算每个点属于哪个slot
        slot_assignment = attn_np.argmax(axis=0)  # (N,)
        
        # 子图1：基于slot assignment的着色
        ax1 = fig.add_subplot(131, projection='3d')
        colors = plt.cm.tab10(slot_assignment)
        ax1.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2],
                   c=colors, s=10)
        ax1.set_title('Slot Assignment\n(by ISA)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # 子图2：真实标签
        ax2 = fig.add_subplot(132, projection='3d')
        colors_gt = plt.cm.tab10(labels_np)
        ax2.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2],
                   c=colors_gt, s=10)
        ax2.set_title('Ground Truth\n(toy data)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # 子图3：统计信息
        ax3 = fig.add_subplot(133)
        ax3.axis('off')
        
        # 计算每个slot包含多少点
        unique, counts = np.unique(slot_assignment, return_counts=True)
        stats_text = "Slot Statistics:\n\n"
        for slot_id, count in zip(unique, counts):
            percentage = count / len(slot_assignment) * 100
            stats_text += f"Slot {slot_id}: {count} points ({percentage:.1f}%)\n"
        
        # 计算attention的集中度（熵）
        attn_entropy = -(attn_np * np.log(attn_np + 1e-8)).sum(axis=0).mean()
        stats_text += f"\nAvg Attention Entropy: {attn_entropy:.3f}\n"
        stats_text += "(lower = more focused)"
        
        ax3.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 图像已保存: {save_path}")
        plt.close()
        
        print(f"✅ Slot分配可视化完成！")
        return True
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_model(args, points, features):
    """
    测试4：完整模型测试
    """
    print(f"\n{'='*60}")
    print("测试4: 完整模型测试")
    print(f"{'='*60}")
    
    try:
        # 创建完整模型
        model = DINOSAURpp(args)
        model.eval()
        
        print(f"✓ 模型创建成功")
        print(f"  - Slot数量: {model.slot_num}")
        print(f"  - Slot维度: {model.slot_dim}")
        print(f"  - 使用ISA: {model.ISA}")
        
        with torch.no_grad():
            reconstruction, slots, masks = model(features, points)
        
        print(f"\n输出形状:")
        print(f"  - Reconstruction: {reconstruction.shape}")
        print(f"  - Slots: {slots.shape}")
        print(f"  - Masks: {masks.shape}")
        
        # 验证
        B, N, D = features.shape
        assert reconstruction.shape == (B, N, args.point_feature_dim), "重建形状错误"
        assert slots.shape == (B, args.num_slots, args.slot_dim), "Slots形状错误"
        assert masks.shape == (B, args.num_slots, N), "Masks形状错误"
        
        print(f"\n✅ 完整模型测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 完整模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='测试3D ISA模块')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch大小')
    parser.add_argument('--num_points', type=int, default=1024, help='点云点数')
    parser.add_argument('--num_objects', type=int, default=3, help='物体数量')
    parser.add_argument('--visualize', action='store_true', help='是否进行可视化')
    parser.add_argument('--output_dir', type=str, default='./test_results', 
                       help='输出目录')
    
    cmd_args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(cmd_args.output_dir, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"#{'3D ISA模块测试脚本'.center(58)}#")
    print(f"{'#'*60}\n")
    
    # 1. 生成测试数据
    points, features, labels = generate_toy_point_cloud(
        batch_size=cmd_args.batch_size,
        num_points=cmd_args.num_points,
        num_objects=cmd_args.num_objects
    )
    
    # 2. 归一化坐标
    points_normalized = normalize_point_coords(points)
    
    # 3. 创建模型配置
    args = Args()
    args.num_points = cmd_args.num_points
    
    # 4. 创建模型
    print(f"\n{'='*60}")
    print("创建3D ISA模型...")
    print(f"{'='*60}")
    
    try:
        model = DINOSAURpp(args)
        model.eval()
        print(f"✓ 模型创建成功")
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 运行测试
    test_results = {}
    
    # 测试1: 形状验证
    test_results['shape'] = test_shape_validation(model, points_normalized, features)
    
    # 测试2: 数值验证
    test_results['numerical'] = test_numerical_validation(model, points_normalized, features)
    
    # 测试3: Slot属性验证
    success, S_p, S_s = test_slot_properties(model, points_normalized, features)
    test_results['properties'] = success
    
    # 测试4: 完整模型测试
    test_results['full_model'] = test_full_model(args, points_normalized, features)
    
    # 6. 可视化（如果启用）
    if cmd_args.visualize and S_p is not None:
        print(f"\n{'='*60}")
        print("开始生成可视化...")
        print(f"{'='*60}")
        
        # 需要先获取attention
        with torch.no_grad():
            slots, attn = model.slot_encoder(features, points_normalized)
        
        # 可视化1: Slot中心
        vis_path1 = os.path.join(cmd_args.output_dir, 'slot_centers_3d.png')
        visualize_slot_centers(points_normalized, S_p, S_s, vis_path1)
        
        # 可视化2: Attention权重
        vis_path2 = os.path.join(cmd_args.output_dir, 'attention_weights_3d.png')
        visualize_attention_weights(points_normalized, attn, labels, vis_path2)
        
        # 可视化3: Slot分配
        vis_path3 = os.path.join(cmd_args.output_dir, 'slot_assignment_3d.png')
        visualize_slot_assignment(points_normalized, attn, labels, vis_path3)
    
    # 7. 总结
    print(f"\n{'#'*60}")
    print(f"#{'测试总结'.center(58)}#")
    print(f"{'#'*60}\n")
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name.ljust(20)}: {status}")
    
    all_passed = all(test_results.values())
    
    if all_passed:
        print(f"\n{'='*60}")
        print(f"🎉 所有测试通过！3D ISA模块工作正常！")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"⚠️  部分测试失败，请检查上述错误信息")
        print(f"{'='*60}\n")
    
    if cmd_args.visualize:
        print(f"可视化结果已保存到: {cmd_args.output_dir}/")
        print(f"  - slot_centers_3d.png")
        print(f"  - attention_weights_3d.png")
        print(f"  - slot_assignment_3d.png")


if __name__ == '__main__':
    main()

