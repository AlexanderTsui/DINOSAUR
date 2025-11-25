"""
PointBERT + DINOSAUR 模型封装

架构:
1. PointBERT Extractor (冻结) - 提取超点级特征
2. Feature Projector (可训练) - 384维 → 768维
3. DINOSAUR ISA (可训练) - Slot Attention
"""

import torch
import torch.nn as nn
import sys
import os

# 添加PointBERT路径
current_dir = os.path.dirname(os.path.abspath(__file__))
pointbert_dir = os.path.join(current_dir, '../../PointBERT')
if pointbert_dir not in sys.path:
    sys.path.insert(0, pointbert_dir)


class FeatureProjector(nn.Module):
    """
    特征投影层: PointBERT输出 → DINOSAUR输入
    384维 → 768维
    """
    def __init__(self, in_dim=384, out_dim=768):
        super().__init__()
        
        mid_dim = (in_dim + out_dim) // 2  # 576
        
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
        return self.projector(x)


class PointBERTDINOSAUR(nn.Module):
    """
    完整模型封装: PointBERT + Projector + DINOSAUR
    
    组件:
    1. PointBERT Extractor (冻结)
    2. Feature Projector (可训练)
    3. DINOSAUR ISA (可训练)
    
    输入: xyzrgb_batch (B, 8192, 6)
    输出: reconstruction, slots, masks, sp_feats_proj
    """
    
    def __init__(self, pointbert_extractor, projector, dinosaur):
        """
        参数:
            pointbert_extractor: PointBERTExtractor实例
            projector: FeatureProjector实例
            dinosaur: DINOSAURpp实例
        """
        super().__init__()
        
        self.pointbert = pointbert_extractor
        self.projector = projector
        self.dinosaur = dinosaur
        
        # 冻结PointBERT
        for param in self.pointbert.parameters():
            param.requires_grad = False
        self.pointbert.eval()
        
        print("[PointBERTDINOSAUR] 模型初始化完成")
        print(f"  - PointBERT: 冻结")
        print(f"  - Projector: 可训练")
        print(f"  - DINOSAUR: 可训练")
    
    def normalize_coords(self, coords):
        """
        将超点坐标归一化到[-1, 1]
        
        参数:
            coords: (B, N, 3) 超点中心坐标
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
    
    def forward(self, xyzrgb_batch):
        """
        前向传播
        
        参数:
            xyzrgb_batch: (B, 8192, 6) - xyz + rgb合并
        
        返回:
            reconstruction: (B, 512, 768) - 重建特征
            slots: (B, num_slots, slot_dim) - slot表示
            masks: (B, num_slots, 512) - 分割mask
            sp_feats_proj: (B, 512, 768) - 投影后的特征（用于loss）
        """
        device = xyzrgb_batch.device
        
        # 1. PointBERT提取超点特征 (冻结，无梯度)
        with torch.no_grad():
            # superpoint_features: (B, 512, 384)
            # centers: (B, 512, 3)
            superpoint_features, centers = self.pointbert(xyzrgb_batch)
        
        # 确保特征在正确的设备上
        superpoint_features = superpoint_features.to(device)
        centers = centers.to(device)
        
        # 2. 特征投影 384 → 768
        sp_feats_proj = self.projector(superpoint_features)  # (B, 512, 768)
        
        # 3. 归一化超点坐标到[-1, 1]
        sp_coords = self.normalize_coords(centers)  # (B, 512, 3)
        
        # 4. DINOSAUR前向传播
        reconstruction, slots, masks = self.dinosaur(
            sp_feats_proj,  # features
            sp_coords       # point_coords
        )
        
        return reconstruction, slots, masks, sp_feats_proj
    
    def unfreeze_pointbert(self):
        """解冻PointBERT（可选微调）"""
        for param in self.pointbert.parameters():
            param.requires_grad = True
        self.pointbert.train()
        print("[PointBERTDINOSAUR] PointBERT已解冻")
    
    def get_trainable_params(self):
        """获取可训练参数"""
        params = []
        params += list(self.projector.parameters())
        params += list(self.dinosaur.parameters())
        return params


def create_pointbert_dinosaur_model(config, device='cuda'):
    """
    工厂函数：创建完整的PointBERT+DINOSAUR模型
    
    参数:
        config: 配置字典
        device: 设备
    
    返回:
        model: PointBERTDINOSAUR实例
    """
    # 导入PointBERT
    sys.path.insert(0, pointbert_dir)
    from extractor import PointBERTExtractor
    
    # 导入DINOSAUR（使用相对导入）
    from .model import DINOSAURpp
    from types import SimpleNamespace
    
    print("=" * 60)
    print("创建 PointBERT + DINOSAUR 模型")
    print("=" * 60)
    
    # 1. 创建PointBERT Extractor
    print("\n[1/3] 加载 PointBERT...")
    pointbert_config_path = config['data']['pointbert_config']
    pointbert_checkpoint = config['data'].get('pointbert_checkpoint', None)
    
    pointbert_extractor = PointBERTExtractor(
        config_path=pointbert_config_path,
        pretrained_path=pointbert_checkpoint,
        device=device
    )
    print("✓ PointBERT 加载完成")
    
    # 2. 创建Feature Projector
    print("\n[2/3] 创建 Feature Projector...")
    projector = FeatureProjector(
        in_dim=config['model']['pointbert_dim'],
        out_dim=config['model']['din_feature_dim']
    )
    print(f"✓ Projector: {config['model']['pointbert_dim']}D → {config['model']['din_feature_dim']}D")
    
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
            self.token_num = config['model']['token_num']
            self.num_points = config['model']['num_points']
            self.point_feature_dim = config['model']['din_feature_dim']
    
    args = Args()
    dinosaur = DINOSAURpp(args)
    print(f"✓ DINOSAUR: {args.num_slots} slots, {args.slot_dim}D")
    
    # 4. 封装完整模型
    print("\n组装完整模型...")
    model = PointBERTDINOSAUR(pointbert_extractor, projector, dinosaur)
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
    print("测试 PointBERT + DINOSAUR Wrapper")
    print("=" * 60)
    
    # 加载配置
    config_path = '../config/config_train_pointbert.yaml'
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        print("请先创建配置文件")
        exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_pointbert_dinosaur_model(config, device=device)
    model.eval()
    
    # 测试前向传播
    print("\n测试前向传播...")
    batch_size = 2
    test_input = torch.randn(batch_size, 8192, 6).to(device)
    
    print(f"输入形状: {test_input.shape}")
    
    with torch.no_grad():
        reconstruction, slots, masks, sp_feats_proj = model(test_input)
    
    print("\n输出形状:")
    print(f"  - reconstruction: {reconstruction.shape}")
    print(f"  - slots: {slots.shape}")
    print(f"  - masks: {masks.shape}")
    print(f"  - sp_feats_proj: {sp_feats_proj.shape}")
    
    print("\n" + "=" * 60)
    print("测试完成！模型可以正常使用")
    print("=" * 60)

