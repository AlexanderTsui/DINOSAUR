"""
改进DINOSAUR：添加对比学习损失
目标：让slots更好地区分物体vs背景
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveDINOSAURLoss(nn.Module):
    """
    在DINOSAUR基础上添加对比学习损失

    核心思想：
    1. Slot内特征应该相似（物体内一致性）
    2. 不同slot的特征应该不同（物体间区分性）
    3. 前景slot应该远离背景特征分布
    """

    def __init__(self, temperature=0.07, bg_slot_index=0):
        super().__init__()
        self.temperature = temperature
        self.bg_slot_index = bg_slot_index  # Two-Stage模式下，slot 0是背景

    def compute_slot_prototypes(self, features, masks):
        """
        计算每个slot的原型特征

        Args:
            features: (B, N, D) - 点特征
            masks: (B, S, N) - slot masks（softmax后）
        Returns:
            prototypes: (B, S, D) - 每个slot的原型
        """
        B, S, N = masks.shape
        D = features.shape[-1]

        # 加权聚合：每个slot的特征 = Σ(mask * features)
        masks_expanded = masks.unsqueeze(-1)  # (B, S, N, 1)
        features_expanded = features.unsqueeze(1)  # (B, 1, N, D)

        weighted_feats = masks_expanded * features_expanded  # (B, S, N, D)
        prototypes = weighted_feats.sum(dim=2)  # (B, S, D)

        # 归一化（避免被mask sum影响尺度）
        mask_sums = masks.sum(dim=2, keepdim=True).clamp(min=1e-6)  # (B, S, 1)
        prototypes = prototypes / mask_sums  # (B, S, D)

        return F.normalize(prototypes, dim=-1)  # L2归一化

    def intra_slot_compactness(self, features, masks, prototypes):
        """
        Slot内紧致性：鼓励slot内points特征接近prototype

        Args:
            features: (B, N, D)
            masks: (B, S, N)
            prototypes: (B, S, D)
        Returns:
            loss: scalar
        """
        B, S, N = masks.shape

        # 归一化特征
        features_norm = F.normalize(features, dim=-1)  # (B, N, D)

        # 计算每个point到所属slot prototype的相似度
        features_exp = features_norm.unsqueeze(1)  # (B, 1, N, D)
        prototypes_exp = prototypes.unsqueeze(2)  # (B, S, 1, D)

        # 余弦相似度
        similarities = (features_exp * prototypes_exp).sum(dim=-1)  # (B, S, N)

        # 加权损失：mask越大的点权重越高
        weighted_sim = (masks * similarities).sum(dim=(1, 2))  # (B,)
        mask_sum = masks.sum(dim=(1, 2)).clamp(min=1e-6)  # (B,)

        # 损失：最大化相似度 = 最小化 -similarity
        loss = -(weighted_sim / mask_sum).mean()

        return loss

    def inter_slot_separation(self, prototypes, exclude_bg=True):
        """
        Slot间分离性：鼓励不同slots的prototypes尽量不同

        Args:
            prototypes: (B, S, D)
            exclude_bg: 是否排除背景slot（避免强制背景与前景分离）
        Returns:
            loss: scalar
        """
        B, S, D = prototypes.shape

        if exclude_bg and S > 1:
            # 只计算前景slots（slot 1~S-1）之间的相似度
            fg_prototypes = prototypes[:, 1:, :]  # (B, S-1, D)
        else:
            fg_prototypes = prototypes

        # 计算pairwise余弦相似度矩阵
        # (B, S', D) x (B, D, S') -> (B, S', S')
        sim_matrix = torch.bmm(fg_prototypes, fg_prototypes.transpose(1, 2))

        # 只取上三角（避免重复计算和对角线）
        S_fg = fg_prototypes.shape[1]
        mask = torch.triu(torch.ones(S_fg, S_fg), diagonal=1).bool().to(sim_matrix.device)

        # 提取上三角相似度
        upper_tri_sim = sim_matrix[:, mask]  # (B, S'*(S'-1)/2)

        # 损失：最小化slot间相似度（鼓励多样性）
        # 使用ReLU: 只惩罚正相关（相似），允许负相关（不同）
        loss = F.relu(upper_tri_sim).mean()

        return loss

    def foreground_background_contrast(self, features, masks, prototypes):
        """
        前景-背景对比：鼓励前景slots远离背景特征分布

        前提：使用Two-Stage DINOSAUR，slot 0是背景

        Args:
            features: (B, N, D)
            masks: (B, S, N)
            prototypes: (B, S, D)
        Returns:
            loss: scalar
        """
        B, S, N = masks.shape

        if S <= 1:
            return torch.tensor(0.0).to(features.device)

        # 背景prototype
        bg_prototype = prototypes[:, self.bg_slot_index, :]  # (B, D)

        # 前景prototypes
        fg_prototypes = torch.cat([
            prototypes[:, :self.bg_slot_index, :],
            prototypes[:, self.bg_slot_index+1:, :]
        ], dim=1)  # (B, S-1, D)

        # 计算前景prototypes与背景prototype的相似度
        bg_proto_exp = bg_prototype.unsqueeze(1)  # (B, 1, D)
        similarities = (fg_prototypes * bg_proto_exp).sum(dim=-1)  # (B, S-1)

        # 损失：最小化前景-背景相似度（鼓励分离）
        loss = F.relu(similarities + 0.1).mean()  # margin=0.1

        return loss

    def forward(self, features, masks, use_two_stage=False):
        """
        完整对比损失

        Args:
            features: (B, N, D) - DINOSAUR输入的点特征（projected）
            masks: (B, S, N) - DINOSAUR输出的slot masks
            use_two_stage: 是否使用Two-Stage（决定是否启用fg-bg对比）
        Returns:
            loss_dict: {
                'compact': intra-slot compactness,
                'separate': inter-slot separation,
                'fg_bg_contrast': foreground-background contrast (if two_stage)
            }
        """
        # 计算slot prototypes
        prototypes = self.compute_slot_prototypes(features, masks)

        # 1. Slot内紧致性
        loss_compact = self.intra_slot_compactness(features, masks, prototypes)

        # 2. Slot间分离性
        loss_separate = self.inter_slot_separation(prototypes, exclude_bg=use_two_stage)

        # 3. 前景-背景对比（仅Two-Stage模式）
        if use_two_stage:
            loss_fg_bg = self.foreground_background_contrast(features, masks, prototypes)
        else:
            loss_fg_bg = torch.tensor(0.0).to(features.device)

        return {
            'compact': loss_compact,
            'separate': loss_separate,
            'fg_bg_contrast': loss_fg_bg
        }


# ==================== 集成到训练 ====================
"""
修改 DINOSAUR/models/losses.py 的 DINOSAURLoss.__init__():

    def __init__(self, loss_cfg):
        super().__init__()
        # ... 现有损失权重 ...

        # 添加对比学习损失权重
        self.w_contrastive_compact = float(weights.get('contrastive_compact', 0.0))
        self.w_contrastive_separate = float(weights.get('contrastive_separate', 0.0))
        self.w_contrastive_fg_bg = float(weights.get('contrastive_fg_bg', 0.0))

        # 初始化对比损失模块
        self.contrastive_loss = ContrastiveDINOSAURLoss(
            temperature=params.get('contrastive_temperature', 0.07),
            bg_slot_index=params.get('bg_slot_index', 0)
        )

修改 DINOSAURLoss.forward():

    def forward(self, reconstruction, target_features, slots, masks,
                sampled_coords=None, use_two_stage=False):
        # ... 现有损失计算 ...

        # 添加对比损失
        if self.w_contrastive_compact > 0 or self.w_contrastive_separate > 0 or self.w_contrastive_fg_bg > 0:
            contrastive_losses = self.contrastive_loss(
                target_features.detach() if self.stop_grad_target else target_features,
                masks,
                use_two_stage=use_two_stage
            )

            loss += self.w_contrastive_compact * contrastive_losses['compact']
            loss += self.w_contrastive_separate * contrastive_losses['separate']
            loss += self.w_contrastive_fg_bg * contrastive_losses['fg_bg_contrast']

            loss_dict['contrastive_compact'] = contrastive_losses['compact'].item()
            loss_dict['contrastive_separate'] = contrastive_losses['separate'].item()
            loss_dict['contrastive_fg_bg'] = contrastive_losses['fg_bg_contrast'].item()

        return loss, loss_dict

配置文件添加（config_train_concerto_scannet.yaml）:

loss:
  weights:
    feat_rec: 1.0
    compact: 0.0
    entropy: 0.0
    min_usage: 0.0
    smooth: 0.0
    cons: 0.0
    diversity: 0.2
    bg_area: 0.0
    # 新增：对比学习损失
    contrastive_compact: 0.5      # Slot内紧致性
    contrastive_separate: 0.3     # Slot间分离性
    contrastive_fg_bg: 0.2        # 前景-背景对比（Two-Stage专用）

  warmup:
    items:
      # ... 现有warmup配置 ...
      contrastive_compact:
        enabled: True
        start_epoch: 20
        warmup_epochs: 30
        start_weight: 0.0
      contrastive_separate:
        enabled: True
        start_epoch: 20
        warmup_epochs: 30
        start_weight: 0.0
      contrastive_fg_bg:
        enabled: True
        start_epoch: 30
        warmup_epochs: 30
        start_weight: 0.0

  params:
    # ... 现有params ...
    contrastive_temperature: 0.07
    bg_slot_index: 0
"""
