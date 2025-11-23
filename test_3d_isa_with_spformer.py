"""
3D ISA模块测试脚本 (集成 SPFormer)

功能流程:
1. 生成玩具点云数据 (xyz + rgb)
2. 使用 SPFormer 提取超点级特征
3. 将超点特征输入 DINOSAUR ISA 模块
4. 可视化 Slot Assignment 结果

使用方法:
    直接运行此脚本，参数在 TestConfig 中修改
    conda activate PointClouds
    python test_3d_isa_with_spformer.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import gorilla
from torch_scatter import scatter_mean, scatter_max

# === 路径设置 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 添加 SPFormer 路径
spformer_path = os.path.abspath(os.path.join(current_dir, '../SPFormer'))
sys.path.insert(0, spformer_path)
lib_path = os.path.join(spformer_path, 'spformer', 'lib')
sys.path.insert(0, lib_path)

print(f"[Info] SPFormer路径: {spformer_path}")
print(f"[Info] Lib路径: {lib_path}")

# 导入 DINOSAUR 模型
try:
    from models.model import DINOSAURpp
except ImportError:
    sys.path.append(os.path.join(current_dir, 'models'))
    from model import DINOSAURpp

# 导入 SPFormer 模型
try:
    import pointgroup_ops
    print(f"[Info] pointgroup_ops加载自: {pointgroup_ops.__file__}")
except ImportError as e:
    print(f"❌ 无法导入 pointgroup_ops: {e}")
    print("请确保已编译 SPFormer 的 C++ 扩展")
    sys.exit(1)

try:
    from spformer.model import SPFormer
    from spformer.utils import get_root_logger
    import spconv.pytorch as spconv
    print("[Info] SPFormer 模块导入成功")
except ImportError as e:
    print(f"❌ 无法导入 SPFormer 模块: {e}")
    sys.exit(1)


class TestConfig:
    """测试配置参数"""
    def __init__(self):
        # === Slot Attention 参数 ===
        self.num_slots = 7           # Slot数量
        self.slot_dim = 256          # Slot特征维度
        self.slot_att_iter = 3       # 迭代次数
        self.query_opt = True        # 是否优化Query
        self.ISA = True              # 使用ISA (3D位置编码)
        
        # === 点云数据参数 ===
        self.num_points = 2000       # 点云点数
        self.batch_size = 1          # Batch大小 (SPFormer单样本)
        self.num_objects = 3         # 模拟物体数量
        self.n_superpoints = 50      # 超点数量
        
        # === 特征维度 ===
        self.din_input_dim = 768     # DINOSAUR输入维度
        
        # === 输出配置 ===
        self.visualize = True
        self.output_dir = './visualization/test_results_3d_spformer'


class TestSPFormerExtractor:
    """SPFormer特征提取器 (测试模式 - 无预训练权重)"""
    
    def __init__(self, config_path, device='cuda'):
        self.device = device
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
        
        self.cfg = gorilla.Config.fromfile(config_path)
        
        print(f"\n[Info] 构建 SPFormer 模型 (测试模式 - 随机权重)...")
        self.model = SPFormer(**self.cfg.model).to(device)
        self.model.eval()
        
        # 获取输出特征维度
        self.output_dim = self.cfg.model.decoder.hidden_dim
        print(f"  - SPFormer 输出维度: {self.output_dim}")
    
    def generate_superpoints(self, xyz, rgb, n_clusters=50):
        """
        使用层次聚类生成超点
        """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.neighbors import kneighbors_graph
        
        print(f"\n[步骤1] 生成超点 (目标数量: {n_clusters})...")
        
        # 构建KNN图保证几何连续性
        connectivity = kneighbors_graph(xyz, n_neighbors=10, include_self=False)
        
        # 层次聚类
        cluster = AgglomerativeClustering(
            n_clusters=n_clusters, 
            connectivity=connectivity, 
            linkage='ward'
        )
        
        # 使用 XYZ + RGB 进行聚类
        labels = cluster.fit_predict(np.concatenate([xyz, rgb], axis=1))
        
        unique_labels = len(np.unique(labels))
        print(f"  ✓ 生成了 {unique_labels} 个超点")
        
        return labels
    
    def prepare_batch(self, xyz, rgb, superpoints):
        """
        准备 SPFormer 输入
        """
        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz).float()
        feat = torch.from_numpy(rgb).float()
        superpoint = torch.from_numpy(superpoints).long()
        
        # Batch偏移量
        batch_offsets = torch.tensor([0, superpoint.max().item() + 1], dtype=torch.int)
        
        # 体素配置
        voxel_cfg = self.cfg.data.test.voxel_cfg
        scale = voxel_cfg.scale
        
        # 缩放和体素化
        coord_float_scaled = coord_float * scale
        coord_float_scaled -= coord_float_scaled.min(0)[0]
        coord_long = coord_float_scaled.long()
        
        # 添加batch索引
        coords_with_batch = torch.cat([
            torch.LongTensor(coord_long.shape[0], 1).fill_(0), 
            coord_long
        ], 1)
        
        # 拼接特征 [RGB + XYZ]
        feats = torch.cat((feat, coord_float_scaled), dim=1)
        
        # 体素化
        spatial_shape_clip = np.clip(
            (coords_with_batch.max(0)[0][1:] + 1).numpy(), 
            voxel_cfg.spatial_shape[0], 
            None
        )
        
        voxel_coords, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(
            coords_with_batch, 1, 4
        )
        
        return {
            'voxel_coords': voxel_coords.to(self.device),
            'p2v_map': p2v_map.to(self.device),
            'v2p_map': v2p_map.to(self.device),
            'spatial_shape': spatial_shape_clip,
            'feats': feats.to(self.device),
            'superpoints': superpoint.to(self.device),
            'batch_offsets': batch_offsets.to(self.device)
        }
    
    def extract(self, xyz, rgb, superpoints):
        """
        执行特征提取
        
        Returns:
            point_features: (N, D) - 点级特征
            sp_feats: (K, D) - 超点级特征
        """
        print(f"\n[步骤2] SPFormer 特征提取...")
        
        batch = self.prepare_batch(xyz, rgb, superpoints)
        batch_size = len(batch['batch_offsets']) - 1
        
        with torch.no_grad():
            # 体素化特征
            voxel_feats = pointgroup_ops.voxelization(
                batch['feats'], 
                batch['v2p_map']
            )
            
            # 构建稀疏张量
            input_tensor = spconv.SparseConvTensor(
                voxel_feats, 
                batch['voxel_coords'].int(), 
                batch['spatial_shape'], 
                batch_size
            )
            
            # U-Net 前向传播
            x = self.model.input_conv(input_tensor)
            x, _ = self.model.unet(x)
            x = self.model.output_layer(x)
            
            # 映射回点
            p2v_map = batch['p2v_map'].long()
            
            # 处理无效索引
            if p2v_map.min() < 0:
                valid_mask = p2v_map >= 0
                point_features = torch.zeros(
                    (p2v_map.shape[0], x.features.shape[1]), 
                    device=self.device, 
                    dtype=x.features.dtype
                )
                if valid_mask.any():
                    point_features[valid_mask] = x.features[p2v_map[valid_mask]]
            else:
                point_features = x.features[p2v_map]
            
            # 超点池化
            if self.model.pool == 'mean':
                sp_feats = scatter_mean(point_features, batch['superpoints'], dim=0)
            else:
                sp_feats, _ = scatter_max(point_features, batch['superpoints'], dim=0)
            
            print(f"  ✓ 点特征: {point_features.shape}")
            print(f"  ✓ 超点特征: {sp_feats.shape}")
            
            return point_features, sp_feats


def generate_toy_data_xyzrgb(config):
    """
    生成玩具点云数据 (XYZ + RGB)
    """
    print(f"\n{'='*60}")
    print("生成玩具点云数据 (XYZ + RGB)...")
    print(f"{'='*60}")
    
    points_per_object = config.num_points // config.num_objects
    xyz_list = []
    rgb_list = []
    
    for obj_id in range(config.num_objects):
        # 每个物体是一个3D高斯簇
        center = np.random.randn(3) * 2
        scale = np.random.rand() * 0.5 + 0.3
        
        obj_xyz = np.random.randn(points_per_object, 3) * scale + center
        
        # 生成不同颜色
        obj_rgb = np.random.rand(points_per_object, 3) * 0.2
        obj_rgb[:, obj_id % 3] += 0.8  # 主色调
        
        xyz_list.append(obj_xyz)
        rgb_list.append(obj_rgb)
    
    # 组合
    xyz = np.concatenate(xyz_list, axis=0)
    rgb = np.concatenate(rgb_list, axis=0)
    
    # 随机打乱
    idx = np.random.permutation(len(xyz))
    xyz = xyz[idx]
    rgb = rgb[idx]
    
    # RGB归一化到 [-1, 1] (ScanNet标准)
    rgb = (rgb - 0.5) * 2
    
    print(f"✓ 点云形状: {xyz.shape}")
    print(f"✓ RGB形状: {rgb.shape}")
    print(f"✓ 坐标范围: [{xyz.min():.2f}, {xyz.max():.2f}]")
    
    return xyz.astype(np.float32), rgb.astype(np.float32)


def compute_superpoint_centers(xyz, superpoints):
    """
    计算超点中心坐标
    
    Args:
        xyz: (N, 3) numpy array
        superpoints: (N,) numpy array
    
    Returns:
        sp_coords: (K, 3) tensor - 超点中心
    """
    print(f"\n[步骤3] 计算超点中心坐标...")
    
    xyz_tensor = torch.from_numpy(xyz).float()
    superpoints_tensor = torch.from_numpy(superpoints).long()
    
    # 使用 scatter_mean 计算中心
    sp_coords = scatter_mean(xyz_tensor, superpoints_tensor, dim=0)
    
    print(f"  ✓ 超点中心: {sp_coords.shape}")
    
    return sp_coords


def normalize_coords(coords):
    """
    归一化坐标到 [-1, 1]
    """
    coords_min = coords.min(dim=0, keepdim=True)[0]
    coords_max = coords.max(dim=0, keepdim=True)[0]
    
    normalized = (coords - coords_min) / (coords_max - coords_min + 1e-8)
    normalized = normalized * 2 - 1
    
    return normalized


def visualize_results(xyz, rgb, superpoints, slot_masks, save_dir):
    """
    三视图可视化: 原始点云 / 超点分割 / Slot分配
    """
    print(f"\n{'='*60}")
    print("生成可视化...")
    print(f"{'='*60}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 转换为 numpy
    if torch.is_tensor(xyz):
        xyz = xyz.cpu().numpy()
    if torch.is_tensor(superpoints):
        superpoints = superpoints.cpu().numpy()
    if torch.is_tensor(slot_masks):
        slot_masks = slot_masks.detach().cpu().numpy()
    
    # Slot分配: 从超点级映射回点级
    sp_slot_ids = slot_masks[0].argmax(axis=0)  # (K,)
    point_slot_ids = sp_slot_ids[superpoints]   # (N,)
    
    num_slots = slot_masks.shape[1]
    
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
    
    # 子图1: 原始点云 (RGB)
    ax1 = fig.add_subplot(131, projection='3d')
    rgb_vis = np.clip((rgb + 1) / 2, 0, 1)  # 从 [-1,1] 转换到 [0,1]
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=2, c=rgb_vis, alpha=0.5)
    ax1.set_title('Input Point Cloud (RGB)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 子图2: 超点分割
    ax2 = fig.add_subplot(132, projection='3d')
    unique_sp = np.unique(superpoints)
    sp_colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(unique_sp)))
    np.random.shuffle(sp_colors)
    point_sp_colors = sp_colors[superpoints]
    ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=2, c=point_sp_colors, alpha=0.5)
    ax2.set_title(f'Superpoints (K={len(unique_sp)})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # 子图3: Slot分配
    ax3 = fig.add_subplot(133, projection='3d')
    point_slot_colors = colors_lookup[point_slot_ids]
    ax3.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=2, c=point_slot_colors, alpha=0.5)
    ax3.set_title(f'Slot Assignment (Total {num_slots} Slots)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'Slot {i}',
               markerfacecolor=colors_lookup[i], markersize=8)
        for i in range(num_slots)
    ]
    ax3.legend(handles=legend_elements, loc='upper left', 
               bbox_to_anchor=(1.05, 1.0), title="Slots")
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'visualization_spformer_dinosaur.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"✓ 可视化已保存: {save_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] 使用设备: {device}")
    
    config = TestConfig()
    
    print(f"\n{'='*60}")
    print("测试配置:")
    print(f"{'='*60}")
    print(f"  - ISA模式: {config.ISA}")
    print(f"  - Slot数量: {config.num_slots}")
    print(f"  - 超点数量: {config.n_superpoints}")
    print(f"  - 点云点数: {config.num_points}")
    
    # ==========================================
    # 步骤1: 初始化 SPFormer
    # ==========================================
    print(f"\n{'='*60}")
    print("初始化 SPFormer...")
    print(f"{'='*60}")
    
    spf_config_path = os.path.join(spformer_path, 'configs/spf_scannet.yaml')
    try:
        extractor = TestSPFormerExtractor(spf_config_path, device=device)
    except Exception as e:
        print(f"❌ SPFormer初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ==========================================
    # 步骤2: 生成数据
    # ==========================================
    xyz, rgb = generate_toy_data_xyzrgb(config)
    
    # ==========================================
    # 步骤3: 生成超点
    # ==========================================
    superpoints = extractor.generate_superpoints(
        xyz, rgb, 
        n_clusters=config.n_superpoints
    )
    
    # ==========================================
    # 步骤4: 提取超点特征
    # ==========================================
    try:
        point_feats, sp_feats = extractor.extract(xyz, rgb, superpoints)
    except Exception as e:
        print(f"❌ 特征提取失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ==========================================
    # 步骤5: 计算超点中心并归一化
    # ==========================================
    sp_coords = compute_superpoint_centers(xyz, superpoints).to(device)
    sp_coords_norm = normalize_coords(sp_coords)
    
    print(f"  ✓ 归一化后坐标范围: [{sp_coords_norm.min():.3f}, {sp_coords_norm.max():.3f}]")
    
    # ==========================================
    # 步骤6: 适配 DINOSAUR 输入
    # ==========================================
    print(f"\n{'='*60}")
    print("适配 DINOSAUR 输入...")
    print(f"{'='*60}")
    
    # 特征维度投影
    if sp_feats.shape[1] != config.din_input_dim:
        print(f"  - 特征维度不匹配: {sp_feats.shape[1]} → {config.din_input_dim}")
        print(f"  - 应用线性投影层...")
        projector = nn.Linear(sp_feats.shape[1], config.din_input_dim).to(device)
        sp_feats_proj = projector(sp_feats)
    else:
        sp_feats_proj = sp_feats
    
    # 添加batch维度
    din_inputs = sp_feats_proj.unsqueeze(0)  # (1, K, 768)
    din_coords = sp_coords_norm.unsqueeze(0)  # (1, K, 3)
    
    print(f"  ✓ DINOSAUR输入特征: {din_inputs.shape}")
    print(f"  ✓ DINOSAUR输入坐标: {din_coords.shape}")
    
    # 更新配置
    config.token_num = din_inputs.shape[1]
    config.num_points = din_inputs.shape[1]
    
    # ==========================================
    # 步骤7: 运行 DINOSAUR (ISA)
    # ==========================================
    print(f"\n{'='*60}")
    print("运行 DINOSAUR (ISA)...")
    print(f"{'='*60}")
    
    try:
        dinosaur = DINOSAURpp(config).to(device)
        dinosaur.eval()
        print("✓ DINOSAUR模型初始化成功")
        
        with torch.no_grad():
            reconstruction, slots, masks = dinosaur(din_inputs, din_coords)
        
        print(f"\n输出形状:")
        print(f"  - Reconstruction: {reconstruction.shape}")
        print(f"  - Slots: {slots.shape}")
        print(f"  - Masks: {masks.shape}")
        
        print(f"\n✅ DINOSAUR推理完成!")
        
    except Exception as e:
        print(f"❌ DINOSAUR运行失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ==========================================
    # 步骤8: 可视化
    # ==========================================
    if config.visualize:
        visualize_results(
            xyz, rgb, 
            superpoints, 
            masks, 
            config.output_dir
        )
    
    print(f"\n{'='*60}")
    print("🎉 测试流程完成!")
    print(f"{'='*60}")
    print(f"结果保存在: {config.output_dir}/")


if __name__ == '__main__':
    main()

