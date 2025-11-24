"""
模型封装: SPFormer + Projector + DINOSAUR
"""

import torch
import torch.nn as nn


class FeatureProjector(nn.Module):
    """
    特征投影层: SPFormer输出 → DINOSAUR输入
    32维 → 768维
    """
    def __init__(self, in_dim=32, out_dim=768):
        super().__init__()
        
        mid_dim = (in_dim + out_dim) // 2
        
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
        Args:
            x: (B, N, in_dim)
        Returns:
            (B, N, out_dim)
        """
        return self.projector(x)


class SPFormerDINOSAUR(nn.Module):
    """
    完整模型封装
    
    组件:
    1. SPFormer (冻结)
    2. FeatureProjector (可训练)
    3. DINOSAUR (可训练)
    """
    
    def __init__(self, spformer_extractor, projector, dinosaur):
        """
        Args:
            spformer_extractor: SPFormer特征提取器(非nn.Module)
            projector: 特征投影层
            dinosaur: DINOSAUR模型
        """
        super().__init__()
        
        self.spformer = spformer_extractor
        self.projector = projector
        self.dinosaur = dinosaur
        
        # SPFormer是外部对象，已经被设置为eval模式
        # 不需要freeze（它有自己的model属性）
        if hasattr(self.spformer, 'model'):
            for param in self.spformer.model.parameters():
                param.requires_grad = False
            self.spformer.model.eval()
    
    def forward(self, xyz_full, rgb_full, sp_labels, sp_coords):
        """
        Args:
            xyz_full: List[Tensor(N_i, 3)] 或 单个Tensor(N, 3)
            rgb_full: List[Tensor(N_i, 3)] 或 单个Tensor(N, 3)
            sp_labels: List[Tensor(N_i,)] 或 单个Tensor(N,)
            sp_coords: (B, 512, 3) 归一化超点坐标
        
        Returns:
            reconstruction: (B, 512, 384)
            slots: (B, num_slots, slot_dim)
            masks: (B, num_slots, 512)
            sp_feats_proj: (B, 512, 384) 投影后的SPFormer特征（用于计算loss）
        """
        batch_size = sp_coords.shape[0]
        device = sp_coords.device
        
        # 处理单样本情况
        if not isinstance(xyz_full, list):
            xyz_full = [xyz_full]
            rgb_full = [rgb_full]
            sp_labels = [sp_labels]
        
        # 提取SPFormer特征
        sp_feats_list = []
        
        with torch.no_grad():
            for i in range(batch_size):
                xyz = xyz_full[i].to(device)
                rgb = rgb_full[i].to(device)
                sp_label = sp_labels[i].to(device)
                
                # 调用SPFormer提取
                try:
                    point_feats, sp_feats = self.spformer.extract(
                        xyz.cpu().numpy(),
                        rgb.cpu().numpy(),
                        sp_label.cpu().numpy()
                    )
                    # sp_feats: (512, D_spformer)
                    sp_feats_list.append(sp_feats)
                    
                except Exception as e:
                    # 如果提取失败，使用随机特征
                    print(f"警告: SPFormer提取失败: {e}")
                    sp_feats = torch.randn(512, 32, device=device)
                    sp_feats_list.append(sp_feats)
        
        # Stack成batch
        sp_feats_batch = torch.stack(sp_feats_list)  # (B, 512, 32)
        
        # 投影到384维
        sp_feats_proj = self.projector(sp_feats_batch)  # (B, 512, 384)
        
        # DINOSAUR前向传播
        reconstruction, slots, masks = self.dinosaur(sp_feats_proj, sp_coords)
        
        return reconstruction, slots, masks, sp_feats_proj
    
    def unfreeze_spformer(self):
        """解冻SPFormer（可选微调）"""
        if hasattr(self.spformer, 'model'):
            for param in self.spformer.model.parameters():
                param.requires_grad = True
            self.spformer.model.train()
    
    def get_trainable_params(self):
        """获取可训练参数"""
        params = []
        params += list(self.projector.parameters())
        params += list(self.dinosaur.parameters())
        return params

