"""
Mask3D + DINOSAUR 模型封装

架构:
1. Mask3DFeatureExtractor (冻结) - 提取点级特征 f_point (96维)
2. FPS采样 - 从变长输出采样固定数量的超点
3. Feature Projector (可训练) - 96维 → hidden_dim
4. DINOSAUR ISA (可训练) - Slot Attention
"""

import torch
import torch.nn as nn
import sys
import os

# 添加PLM路径
current_dir = os.path.dirname(os.path.abspath(__file__))
plm_dir = os.path.join(current_dir, '../../PLM')
pll_dir = os.path.join(plm_dir, 'pll')

if plm_dir not in sys.path:
    sys.path.insert(0, plm_dir)
if pll_dir not in sys.path:
    sys.path.insert(0, pll_dir)

# 导入FPS采样
try:
    from pointnet2_ops import pointnet2_utils
    HAS_POINTNET2 = True
except ImportError:
    print("[Warning] pointnet2_ops未安装，将使用随机采样代替FPS")
    HAS_POINTNET2 = False


class FeatureProjector(nn.Module):
    """
    特征投影层: Mask3D f_point输出 → DINOSAUR输入
    96维 → hidden_dim (如768)
    
    增加了输入特征归一化，防止极端值导致梯度NaN
    """
    def __init__(self, in_dim=96, out_dim=768):
        super().__init__()
        
        mid_dim = (in_dim + out_dim) // 2
        
        # 输入归一化层，稳定Mask3D特征的数值范围
        self.input_norm = nn.LayerNorm(in_dim)
        
        self.projector = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mid_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
    
    def forward(self, x):
        """
        参数:
            x: (B, N, in_dim)
        返回:
            (B, N, out_dim)
        """
        # 先对输入做归一化，防止Mask3D特征的极端值
        x = self.input_norm(x)
        out = self.projector(x)
        # 输出范围控制，防止极端值传播到下游ISA模块导致梯度爆炸
        out = torch.clamp(out, min=-10, max=10)
        return out


def fps_sample(features, coords, num_points, device):
    """
    使用FPS从变长特征中采样固定数量的点
    
    参数:
        features: (M, C) 特征
        coords: (M, 3) 坐标
        num_points: 采样数量
        device: 设备
    返回:
        sampled_features: (num_points, C)
        sampled_coords: (num_points, 3)
    """
    M = features.shape[0]
    
    if M <= num_points:
        # 点数不足，需要padding（随机重复）
        if M == 0:
            # 完全空的情况，返回零填充
            return (
                torch.zeros(num_points, features.shape[1], device=device),
                torch.zeros(num_points, 3, device=device)
            )
        pad_indices = torch.randint(0, M, (num_points - M,), device=device)
        indices = torch.cat([torch.arange(M, device=device), pad_indices])
    else:
        # FPS采样
        if HAS_POINTNET2:
            coords_batch = coords.unsqueeze(0).float().contiguous()  # (1, M, 3)
            fps_indices = pointnet2_utils.furthest_point_sample(
                coords_batch, num_points
            ).squeeze(0).long()  # (num_points,)
            indices = fps_indices
        else:
            # 随机采样（fallback）
            indices = torch.randperm(M, device=device)[:num_points]
    
    return features[indices], coords[indices]


class Mask3DDINOSAUR(nn.Module):
    """
    完整模型封装: Mask3D + Projector + DINOSAUR
    
    组件:
    1. Mask3DFeatureExtractor (冻结)
    2. FPS采样层
    3. Feature Projector (可训练)
    4. DINOSAUR ISA (可训练)
    
    输入: xyz (B, N, 3), rgb (B, N, 3)
    输出: reconstruction, slots, masks, sp_feats_proj, sampled_coords
    """
    
    def __init__(self, mask3d_extractor, projector, dinosaur, num_superpoints=4096):
        """
        参数:
            mask3d_extractor: Mask3D模型实例
            projector: FeatureProjector实例
            dinosaur: DINOSAURpp实例
            num_superpoints: FPS采样的超点数量
        """
        super().__init__()
        
        self.mask3d = mask3d_extractor
        self.projector = projector
        self.dinosaur = dinosaur
        self.num_superpoints = num_superpoints
        
        # 冻结Mask3D
        for param in self.mask3d.parameters():
            param.requires_grad = False
        self.mask3d.eval()
        
        print("[Mask3DDINOSAUR] 模型初始化完成")
        print(f"  - Mask3D: 冻结")
        print(f"  - 超点数量: {num_superpoints}")
        print(f"  - Projector: 可训练")
        print(f"  - DINOSAUR: 可训练")
    
    def normalize_coords(self, coords):
        """
        将超点坐标归一化到[-1, 1]
        
        参数:
            coords: (B, N, 3) 超点坐标
        返回:
            coords_norm: (B, N, 3) 归一化后的坐标
        """
        B = coords.shape[0]
        coords_norm = []
        
        for i in range(B):
            c = coords[i]  # (N, 3)
            
            # 找到边界
            c_min = c.min(dim=0)[0]  # (3,)
            c_max = c.max(dim=0)[0]  # (3,)
            
            # 归一化到[0, 1]再映射到[-1, 1]
            c_norm = (c - c_min) / (c_max - c_min + 1e-8)
            c_norm = c_norm * 2 - 1
            
            coords_norm.append(c_norm)
        
        return torch.stack(coords_norm, dim=0)  # (B, N, 3)
    
    def extract_and_sample(self, xyz, rgb, device):
        """
        提取Mask3D特征并FPS采样
        
        参数:
            xyz: (B, N, 3) 坐标
            rgb: (B, N, 3) 颜色
            device: 设备
        返回:
            sampled_feats: (B, num_superpoints, 96)
            sampled_coords: (B, num_superpoints, 3)
        """
        # 导入Mask3D的数据准备函数
        from mask3d import prepare_data_tensor
        
        B = xyz.shape[0]
        all_feats = []
        all_coords = []
        
        for i in range(B):
            # 准备单个样本的数据
            xyz_i = xyz[i]  # (N, 3)
            rgb_i = rgb[i]  # (N, 3)
            
            # Mask3D数据准备（体素化）
            data, features, unique_map, inverse_map = prepare_data_tensor(
                [xyz_i], [rgb_i], device=device
            )
            
            # 前向传播获取特征
            with torch.no_grad():
                outputs = self.mask3d.model(data, raw_coordinates=features[:, -3:])
            
            # 提取f_point (backbone_features)
            # backbone_features是一个列表，每个元素对应一个batch样本
            f_point = outputs['backbone_features'][0]  # (M, 96)
            
            # 数值稳定性：裁剪极端值，防止NaN梯度
            f_point = torch.clamp(f_point, min=-100, max=100)
            f_point = torch.nan_to_num(f_point, nan=0.0, posinf=100.0, neginf=-100.0)
            
            # 获取体素化后的坐标
            # 对于单个样本，features的形状是(M, 6)，最后3维是原始坐标
            # 由于我们只处理单个样本，features已经是体素化后的特征
            voxel_coords = features[:, -3:].clone()  # (M, 3) - 体素化后的原始坐标
            
            # 确保f_point和voxel_coords的长度一致
            M = f_point.shape[0]
            if voxel_coords.shape[0] != M:
                # 如果长度不一致，使用f_point的长度
                voxel_coords = voxel_coords[:M]
            
            # FPS采样固定数量的点
            sampled_feats, sampled_coords = fps_sample(
                f_point, voxel_coords, self.num_superpoints, device
            )
            
            all_feats.append(sampled_feats)
            all_coords.append(sampled_coords)
        
        # Stack成batch
        sp_feats = torch.stack(all_feats)      # (B, num_superpoints, 96)
        sp_coords = torch.stack(all_coords)    # (B, num_superpoints, 3)
        
        return sp_feats, sp_coords
    
    def forward(self, xyz, rgb):
        """
        前向传播
        
        参数:
            xyz: (B, N, 3) 坐标
            rgb: (B, N, 3) 颜色 [0, 1]
        
        返回:
            reconstruction: (B, num_superpoints, hidden_dim) - 重建特征
            slots: (B, num_slots, slot_dim) - slot表示
            masks: (B, num_slots, num_superpoints) - 分割mask
            sp_feats_proj: (B, num_superpoints, hidden_dim) - 投影后的特征（用于loss）
            sampled_coords: (B, num_superpoints, 3) - 采样后的坐标（用于可视化）
        """
        device = xyz.device
        
        # 1. Mask3D特征提取 + FPS采样（冻结，无梯度）
        sp_feats, sampled_coords = self.extract_and_sample(xyz, rgb, device)
        
        # 2. 特征投影 96 → hidden_dim
        sp_feats_proj = self.projector(sp_feats)  # (B, num_superpoints, hidden_dim)
        
        # 3. 归一化超点坐标到[-1, 1]
        sp_coords_norm = self.normalize_coords(sampled_coords)  # (B, num_superpoints, 3)
        
        # 4. DINOSAUR前向传播
        # 关键：ISA模块对数值精度敏感，在FP32下执行以避免梯度NaN
        # 使用torch.cuda.amp.autocast(enabled=False)强制FP32
        from torch.cuda.amp import autocast
        with autocast(enabled=False):
            # 确保输入是FP32
            sp_feats_proj_fp32 = sp_feats_proj.float()
            sp_coords_norm_fp32 = sp_coords_norm.float()
            
            reconstruction, slots, masks = self.dinosaur(
                sp_feats_proj_fp32,    # features
                sp_coords_norm_fp32    # point_coords
            )
        
        return reconstruction, slots, masks, sp_feats_proj, sampled_coords
    
    def get_trainable_params(self):
        """获取可训练参数"""
        params = []
        params += list(self.projector.parameters())
        params += list(self.dinosaur.parameters())
        return params


def create_mask3d_dinosaur_model(config, device='cuda'):
    """
    工厂函数：创建完整的Mask3D+DINOSAUR模型
    
    参数:
        config: 配置字典
        device: 设备
    
    返回:
        model: Mask3DDINOSAUR实例
    """
    # 导入Mask3D
    from mask3d import get_model
    
    # 导入DINOSAUR
    from .model import DINOSAURpp
    
    print("=" * 60)
    print("创建 Mask3D + DINOSAUR 模型")
    print("=" * 60)
    
    # 1. 创建Mask3D模型
    print("\n[1/3] 加载 Mask3D...")
    mask3d_checkpoint = config['model']['mask3d_checkpoint']
    mask3d_model = get_model(mask3d_checkpoint)
    mask3d_model = mask3d_model.to(device)
    mask3d_model.eval()
    print(f"✓ Mask3D 加载完成: {mask3d_checkpoint}")
    
    # 2. 创建Feature Projector
    print("\n[2/3] 创建 Feature Projector...")
    projector = FeatureProjector(
        in_dim=config['model']['input_dim'],
        out_dim=config['model']['hidden_dim']
    )
    print(f"✓ Projector: {config['model']['input_dim']}D → {config['model']['hidden_dim']}D")
    
    # 3. 创建DINOSAUR
    print("\n[3/3] 创建 DINOSAUR...")
    
    # 配置DINOSAUR参数
    class Args:
        def __init__(self):
            self.num_slots = config['model']['num_slots']
            self.slot_dim = config['model']['slot_dim']
            self.slot_att_iter = config['model']['slot_att_iter']
            self.query_opt = config['model'].get('query_opt', True)
            self.ISA = config['model']['ISA']
            self.token_num = config['model']['num_superpoints']
            self.num_points = config['model']['num_superpoints']
            self.point_feature_dim = config['model']['hidden_dim']
    
    args = Args()
    dinosaur = DINOSAURpp(args)
    print(f"✓ DINOSAUR: {args.num_slots} slots, {args.slot_dim}D")
    
    # 4. 封装完整模型
    print("\n组装完整模型...")
    model = Mask3DDINOSAUR(
        mask3d_extractor=mask3d_model,
        projector=projector,
        dinosaur=dinosaur,
        num_superpoints=config['model']['num_superpoints']
    )
    model = model.to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "=" * 60)
    print("模型统计:")
    print("=" * 60)
    print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"冻结参数: {total_params - trainable_params:,} ({(total_params - trainable_params)/1e6:.2f}M)")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    """
    测试模型封装
    """
    import yaml
    
    print("=" * 60)
    print("测试 Mask3D + DINOSAUR Wrapper")
    print("=" * 60)
    
    # 加载配置
    config_path = os.path.join(current_dir, '../config/config_train_mask3d.yaml')
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        print("请先创建配置文件")
        exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_mask3d_dinosaur_model(config, device=device)
    model.eval()
    
    # 测试前向传播
    print("\n测试前向传播...")
    batch_size = 2
    num_points = 80000
    test_xyz = torch.randn(batch_size, num_points, 3).to(device)
    test_rgb = torch.rand(batch_size, num_points, 3).to(device)
    
    print(f"输入形状: xyz={test_xyz.shape}, rgb={test_rgb.shape}")
    
    with torch.no_grad():
        reconstruction, slots, masks, sp_feats_proj, sampled_coords = model(test_xyz, test_rgb)
    
    print("\n输出形状:")
    print(f"  - reconstruction: {reconstruction.shape}")
    print(f"  - slots: {slots.shape}")
    print(f"  - masks: {masks.shape}")
    print(f"  - sp_feats_proj: {sp_feats_proj.shape}")
    print(f"  - sampled_coords: {sampled_coords.shape}")
    
    print("\n" + "=" * 60)
    print("测试完成！模型可以正常使用")
    print("=" * 60)

