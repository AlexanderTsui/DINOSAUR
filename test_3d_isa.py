"""
3D ISA模块测试和可视化脚本（可选集成SPFormer）

用途：
1. 验证从2D到3D的修改是否正确（无预训练权重情况下）
2. 可视化slot在3D空间中的分布
3. 检查模型的前向传播、梯度流动和维度匹配
4. 当 config.use_spformer = True 时，调用SPFormer生成超点特征再进入DINOSAUR

使用方法：
    直接运行此脚本：python test_3d_isa.py
    参数可在 main 函数中直接修改（含 use_spformer 开关）
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import importlib

# 添加模型路径
sys.path.append(os.path.dirname(__file__))

try:
    DINOSAURpp = importlib.import_module('models.model').DINOSAURpp
except ModuleNotFoundError:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
    DINOSAURpp = importlib.import_module('model').DINOSAURpp


class TestConfig:
    """测试配置参数"""
    def __init__(self):
        # === 模式选择 ===
        self.use_spformer = False     # 是否走SPFormer→DINOSAUR流程
        
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_superpoints = 50       # 使用SPFormer时的超点数量
        self.spformer_config_path = os.path.join(
            os.path.dirname(__file__), 
            '../SPFormer/configs/spf_scannet.yaml'
        )
        
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


def generate_toy_data_xyzrgb(config):
    """
    生成包含RGB信息的玩具点云，供SPFormer管线测试
    """
    print(f"\n{'='*60}")
    print("生成玩具点云数据 (XYZ + RGB)...")
    print(f"{'='*60}")
    
    points_per_object = config.num_points // config.num_objects
    xyz_list = []
    rgb_list = []
    
    for obj_id in range(config.num_objects):
        center = np.random.randn(3) * 2
        scale = np.random.rand() * 0.5 + 0.3
        obj_xyz = np.random.randn(points_per_object, 3) * scale + center
        
        obj_rgb = np.random.rand(points_per_object, 3) * 0.2
        obj_rgb[:, obj_id % 3] += 0.8  # 简单赋予主色调
        
        xyz_list.append(obj_xyz)
        rgb_list.append(obj_rgb)
    
    xyz = np.concatenate(xyz_list, axis=0)
    rgb = np.concatenate(rgb_list, axis=0)
    
    idx = np.random.permutation(len(xyz))
    xyz = xyz[idx]
    rgb = rgb[idx]
    
    rgb = (rgb - 0.5) * 2  # 归一化到[-1, 1]
    
    print(f"✓ 点云形状: {xyz.shape}")
    print(f"✓ RGB形状: {rgb.shape}")
    print(f"✓ 坐标范围: [{xyz.min():.2f}, {xyz.max():.2f}]")
    
    return xyz.astype(np.float32), rgb.astype(np.float32)


class TestSPFormerExtractor:
    """SPFormer特征提取器（测试模式，无预训练权重）"""
    def __init__(self, config_path, device='cuda'):
        try:
            gorilla = importlib.import_module('gorilla')
            spformer_model = importlib.import_module('spformer.model')
            spconv = importlib.import_module('spconv.pytorch')
            pointgroup_ops = importlib.import_module('pointgroup_ops')
            torch_scatter = importlib.import_module('torch_scatter')
            scatter_mean = getattr(torch_scatter, 'scatter_mean')
            scatter_max = getattr(torch_scatter, 'scatter_max')
        except ImportError as e:
            raise ImportError("SPFormer依赖未安装或未编译，请确认后再开启use_spformer") from e
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"SPFormer配置文件不存在: {config_path}")
        
        self.device = device
        self.gorilla = gorilla
        self.SPFormer = spformer_model.SPFormer
        self.spconv = spconv
        self.pointgroup_ops = pointgroup_ops
        self.scatter_mean = scatter_mean
        self.scatter_max = scatter_max
        
        self.cfg = gorilla.Config.fromfile(config_path)
        
        print(f"\n[SPFormer] 构建模型 (随机权重)...")
        self.model = self.SPFormer(**self.cfg.model).to(device)
        self.model.eval()
        self.output_dim = self.cfg.model.decoder.hidden_dim
        print(f"[SPFormer] 输出维度: {self.output_dim}")
    
    def generate_superpoints(self, xyz, rgb, n_clusters=50):
        """简单层次聚类生成超点"""
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.neighbors import kneighbors_graph
        
        print(f"\n[SPFormer] 生成超点 (K={n_clusters})...")
        
        connectivity = kneighbors_graph(xyz, n_neighbors=10, include_self=False)
        cluster = AgglomerativeClustering(
            n_clusters=n_clusters,
            connectivity=connectivity,
            linkage='ward'
        )
        labels = cluster.fit_predict(np.concatenate([xyz, rgb], axis=1))
        print(f"[SPFormer] 实际生成 {len(np.unique(labels))} 个超点")
        return labels
    
    def prepare_batch(self, xyz, rgb, superpoints):
        """整理SPFormer所需输入"""
        coord_float = torch.from_numpy(xyz).float()
        feat_rgb = torch.from_numpy(rgb).float()
        superpoint = torch.from_numpy(superpoints).long()
        
        voxel_cfg = self.cfg.data.test.voxel_cfg
        scale = voxel_cfg.scale
        
        coord_float_scaled = coord_float * scale
        coord_float_scaled -= coord_float_scaled.min(0)[0]
        coord_long = coord_float_scaled.long()
        
        coords_with_batch = torch.cat([
            torch.zeros(coord_long.shape[0], 1).long(),
            coord_long
        ], dim=1)
        
        feats = torch.cat((feat_rgb, coord_float_scaled), dim=1)
        
        spatial_shape_clip = np.clip(
            (coords_with_batch.max(0)[0][1:] + 1).numpy(),
            voxel_cfg.spatial_shape[0],
            None
        )
        
        voxel_coords, p2v_map, v2p_map = self.pointgroup_ops.voxelization_idx(
            coords_with_batch, 1, 4
        )
        
        batch_offsets = torch.tensor([0, superpoint.max().item() + 1], dtype=torch.int)
        
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
        使用SPFormer提取点级及超点级特征
        Returns:
            point_features: (N, D)
            sp_feats: (K, D)
        """
        batch = self.prepare_batch(xyz, rgb, superpoints)
        batch_size = len(batch['batch_offsets']) - 1
        
        with torch.no_grad():
            voxel_feats = self.pointgroup_ops.voxelization(
                batch['feats'],
                batch['v2p_map']
            )
            
            input_tensor = self.spconv.SparseConvTensor(
                voxel_feats,
                batch['voxel_coords'].int(),
                batch['spatial_shape'],
                batch_size
            )
            
            x = self.model.input_conv(input_tensor)
            x, _ = self.model.unet(x)
            x = self.model.output_layer(x)
            
            p2v_map = batch['p2v_map'].long()
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
            
            if self.model.pool == 'mean':
                sp_feats = self.scatter_mean(point_features, batch['superpoints'], dim=0)
            else:
                sp_feats, _ = self.scatter_max(point_features, batch['superpoints'], dim=0)
        
        print(f"[SPFormer] 点特征: {point_features.shape}")
        print(f"[SPFormer] 超点特征: {sp_feats.shape}")
        return point_features, sp_feats


def compute_superpoint_centers(xyz, superpoints):
    """
    计算每个超点的几何中心
    """
    from torch_scatter import scatter_mean
    
    xyz_tensor = torch.from_numpy(xyz).float()
    sp_tensor = torch.from_numpy(superpoints).long()
    
    sp_coords = scatter_mean(xyz_tensor, sp_tensor, dim=0)
    print(f"[SPFormer] 超点中心: {sp_coords.shape}")
    return sp_coords


def prepare_spformer_inputs(config):
    """
    完整的SPFormer→DINOSAUR输入准备流程
    Returns:
        points: (1, K, 3) torch.FloatTensor
        features: (1, K, D) torch.FloatTensor
        extra_vis: dict，可选可视化信息
    """
    xyz, rgb = generate_toy_data_xyzrgb(config)
    
    extractor = TestSPFormerExtractor(
        config.spformer_config_path,
        device=config.device
    )
    
    superpoints = extractor.generate_superpoints(
        xyz, rgb,
        n_clusters=config.n_superpoints
    )
    
    _, sp_feats = extractor.extract(xyz, rgb, superpoints)
    sp_coords = compute_superpoint_centers(xyz, superpoints)
    
    if sp_feats.shape[1] != config.point_feature_dim:
        print(f"[SPFormer] 特征维度不匹配: {sp_feats.shape[1]} → {config.point_feature_dim}")
        projector = nn.Linear(sp_feats.shape[1], config.point_feature_dim).to(config.device)
        sp_feats = projector(sp_feats)
    
    points = sp_coords.unsqueeze(0).cpu()
    features = sp_feats.unsqueeze(0).detach().cpu()
    
    extra_vis = {
        'xyz': xyz,
        'rgb': rgb,
        'superpoints': superpoints
    }
    
    return points, features, extra_vis


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


def visualize_results(points, masks, slots, save_dir, extra=None):
    """
    生成可视化结果
    """
    print(f"\n{'='*60}")
    print("生成可视化报告")
    print(f"{'='*60}")
    
    extra = extra or {}
    
    # 如果提供了原始xyz/rgb/超点信息，则生成更丰富的可视化
    if extra.get('xyz') is not None and extra.get('superpoints') is not None:
        visualize_spformer_results(extra, masks, save_dir)
        return
    
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


def visualize_spformer_results(extra, slot_masks, save_dir):
    """
    使用SPFormer数据的三视图可视化
    """
    os.makedirs(save_dir, exist_ok=True)
    
    xyz = extra['xyz']
    rgb = extra.get('rgb')
    superpoints = extra['superpoints']
    
    if torch.is_tensor(slot_masks):
        slot_masks = slot_masks.detach().cpu().numpy()
    if torch.is_tensor(superpoints):
        superpoints = superpoints.cpu().numpy()
    
    sp_slot_ids = slot_masks[0].argmax(axis=0)
    point_slot_ids = sp_slot_ids[superpoints]
    num_slots = slot_masks.shape[1]
    
    if num_slots <= 10:
        cmap = plt.get_cmap('tab10')
        colors_lookup = np.array([cmap(i) for i in range(num_slots)])
    elif num_slots <= 20:
        cmap = plt.get_cmap('tab20')
        colors_lookup = np.array([cmap(i) for i in range(num_slots)])
    else:
        cmap = plt.get_cmap('hsv')
        colors_lookup = np.array([cmap(i / num_slots) for i in range(num_slots)])
    
    fig = plt.figure(figsize=(18, 6))
    
    ax1 = fig.add_subplot(131, projection='3d')
    rgb_vis = np.clip((rgb + 1) / 2, 0, 1)
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=2, c=rgb_vis, alpha=0.5)
    ax1.set_title('Input Point Cloud (RGB)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
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
    
    ax3 = fig.add_subplot(133, projection='3d')
    point_slot_colors = colors_lookup[point_slot_ids]
    ax3.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=2, c=point_slot_colors, alpha=0.5)
    ax3.set_title(f'Slot Assignment (Total {num_slots} Slots)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
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
    # 1. 配置
    config = TestConfig()
    extra_vis = {}
    
    print(f"测试配置:")
    print(f"  - ISA模式: {config.ISA}")
    print(f"  - Slot数量: {config.num_slots}")
    print(f"  - 特征维度: {config.point_feature_dim}")
    print(f"  - 使用SPFormer: {config.use_spformer}")
    
    # 2. 准备数据
    if config.use_spformer:
        try:
            points, features, extra_vis = prepare_spformer_inputs(config)
            config.batch_size = 1
            config.num_points = features.shape[1]
            config.token_num = config.num_points
            print(f"  - SPFormer生成的超点数量: {config.num_points}")
        except Exception as e:
            print(f"❌ SPFormer数据准备失败: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
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
            visualize_results(points_norm, masks, slots, config.output_dir, extra_vis)
    else:
        print(f"\n{'='*60}")
        print("⚠️ 测试发现问题，请检查日志。")
        print(f"{'='*60}")

if __name__ == '__main__':
    main()
