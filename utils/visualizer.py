"""
可视化工具
生成训练过程中的图像
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_slot_assignment(xyz, sp_labels, masks, save_path, num_slots=16):
    """
    可视化Slot分配结果（三视图）
    
    Args:
        xyz: (N, 3) numpy array - 点云坐标
        sp_labels: (N,) numpy array - 超点标签
        masks: (S, 512) numpy array - Slot分配
        save_path: 保存路径
        num_slots: Slot数量
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Slot分配: 从超点级映射回点级
    sp_slot_ids = masks.argmax(axis=0)  # (512,)
    point_slot_ids = sp_slot_ids[sp_labels]  # (N,)
    
    # 生成颜色
    if num_slots <= 10:
        cmap = plt.get_cmap('tab10')
        colors_lookup = np.array([cmap(i) for i in range(num_slots)])
    elif num_slots <= 20:
        cmap = plt.get_cmap('tab20')
        colors_lookup = np.array([cmap(i) for i in range(num_slots)])
    else:
        cmap = plt.get_cmap('hsv')
        colors_lookup = np.array([cmap(i / num_slots) for i in range(num_slots)])
    
    # 创建图表
    fig = plt.figure(figsize=(18, 6))
    
    # 子图1: 输入点云（灰色）
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, c='gray', alpha=0.3)
    ax1.set_title('Input Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 子图2: 超点分割
    ax2 = fig.add_subplot(132, projection='3d')
    unique_sp = np.unique(sp_labels)
    sp_colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(unique_sp)))
    np.random.shuffle(sp_colors)
    point_sp_colors = sp_colors[sp_labels]
    ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, c=point_sp_colors, alpha=0.5)
    ax2.set_title(f'Superpoints (K={len(unique_sp)})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # 子图3: Slot分配
    ax3 = fig.add_subplot(133, projection='3d')
    point_slot_colors = colors_lookup[point_slot_ids]
    ax3.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, c=point_slot_colors, alpha=0.5)
    ax3.set_title(f'Slot Assignment ({num_slots} Slots)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_reconstruction_error(xyz, sp_labels, reconstruction, sp_feats_proj, save_path):
    """
    可视化重建误差热力图
    
    Args:
        xyz: (N, 3) numpy array
        sp_labels: (N,) numpy array
        reconstruction: (512, D) numpy array
        sp_feats_proj: (512, D) numpy array
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 计算每个超点的MSE
    mse_per_sp = ((reconstruction - sp_feats_proj) ** 2).mean(axis=1)  # (512,)
    
    # 映射到点
    point_errors = mse_per_sp[sp_labels]  # (N,)
    
    # 可视化
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        c=point_errors,
        cmap='hot',
        s=2,
        vmin=0,
        vmax=np.percentile(point_errors, 95)
    )
    
    ax.set_title('Reconstruction Error (MSE)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.colorbar(scatter, ax=ax, label='MSE')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_slot_statistics(masks, save_path):
    """
    可视化Slot使用统计
    
    Args:
        masks: (S, 512) numpy array
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 计算每个Slot分配到的超点数量
    sp_assignment = masks.argmax(axis=0)  # (512,)
    unique, counts = np.unique(sp_assignment, return_counts=True)
    
    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    slot_ids = np.arange(masks.shape[0])
    slot_counts = np.zeros(masks.shape[0])
    for sid, count in zip(unique, counts):
        slot_counts[sid] = count
    
    bars = ax.bar(slot_ids, slot_counts)
    
    # 着色
    colors = plt.cm.tab20(np.linspace(0, 1, len(slot_ids)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Slot ID')
    ax.set_ylabel('Number of Superpoints')
    ax.set_title('Slot Usage Statistics')
    ax.set_xticks(slot_ids)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

