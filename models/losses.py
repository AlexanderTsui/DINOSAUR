"""
DINOSAUR 3D训练损失函数
包含4项损失:
1. Reconstruction Loss (MSE)
2. Mask Entropy Loss
3. Slot Diversity Loss
4. Mask Uniformity Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOSAURLoss(nn.Module):
    """
    DINOSAUR损失函数
    """
    
    def __init__(self, loss_weights):
        """
        Args:
            loss_weights: dict with keys:
                - reconstruction
                - mask_entropy
                - slot_diversity
                - mask_uniformity  # 新增：mask均匀性损失权重
        """
        super().__init__()
        
        self.w_recon = loss_weights.get('reconstruction', 1.0)
        self.w_entropy = loss_weights.get('mask_entropy', 0.15)
        self.w_diversity = loss_weights.get('slot_diversity', 0.08)
        self.w_uniformity = loss_weights.get('mask_uniformity', 0.3)  # 新增权重
        
        self.mse = nn.MSELoss()
    
    def reconstruction_loss(self, reconstruction, target):
        """
        重建损失 (MSE)
        
        Args:
            reconstruction: (B, N, D)
            target: (B, N, D)
        """
        return self.mse(reconstruction, target.detach())
    
    def mask_entropy_loss(self, masks):
        """
        Mask熵损失 - 鼓励确定性分配
        
        Args:
            masks: (B, S, N) - 每个超点在S个Slot上的分布
        
        计算每个超点的熵，熵越小说明分配越确定
        """
        # masks应该已经在[0,1]且sum(dim=1)=1
        # 为安全起见，重新归一化（防止空slot导致的数值问题）
        masks_sum = masks.sum(dim=1, keepdim=True)
        masks = masks / (masks_sum + 1e-8)
        
        # 过滤掉全为0的mask（空slot或无效点）
        valid_mask = (masks_sum.squeeze(1) > 1e-6)  # (B, N)
        
        # 计算熵: H = -sum(p * log(p))
        # 对每个超点计算其在S个Slot上的分布熵
        entropy_per_point = -(masks * torch.log(masks + 1e-8)).sum(dim=1)  # (B, N)
        
        # 只计算有效点的熵
        if valid_mask.sum() > 0:
            return entropy_per_point[valid_mask].mean()
        else:
            return torch.tensor(0.0, device=masks.device)
    
    def slot_diversity_loss(self, slots):
        """
        Slot多样性损失 - 使用余弦相似度并增加数值稳定处理
        
        Args:
            slots: (B, S, D_slot)
        """
        B, S, D = slots.shape
        
        # L2归一化时显式裁剪范数，避免除零导致的NaN
        slot_norm = slots.norm(dim=-1, keepdim=True)  # (B, S, 1)
        safe_norm = torch.clamp(slot_norm, min=1e-6)
        slots_normalized = slots / safe_norm
        slots_normalized = torch.where(
            slot_norm > 1e-6,
            slots_normalized,
            torch.zeros_like(slots_normalized)
        )
        
        # 计算余弦相似度矩阵并过滤无效值
        cos_sim = torch.matmul(slots_normalized, slots_normalized.transpose(1, 2))  # (B, S, S)
        cos_sim = cos_sim.clamp(-1.0 + 1e-4, 1.0 - 1e-4)
        cos_sim = torch.nan_to_num(cos_sim, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 去掉对角元素，仅约束不同slot之间的相似度
        eye = torch.eye(S, device=slots.device).unsqueeze(0)  # (1, S, S)
        cos_off_diag = cos_sim * (1 - eye)
        
        # 只惩罚正相似度，鼓励slot正交分布
        loss = F.relu(cos_off_diag).pow(2).mean()
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("[警告] slot_diversity_loss: 损失为NaN/Inf，返回零损失")
            return torch.tensor(0.0, device=slots.device, requires_grad=True)
        
        return loss
    
    def mask_uniformity_loss(self, masks):
        """
        Mask均匀性损失 - 防止所有点塌缩到单个slot
        
        Args:
            masks: (B, S, N) - 每个超点在S个Slot上的分布
        
        改进版：同时考虑软分配和硬分配
        - 软分配：计算mask总和的方差
        - 硬分配：计算argmax对应的点数方差
        """
        B, S, N = masks.shape
        
        # 数值稳定性：确保masks非负且有限
        masks = torch.clamp(masks, min=0.0, max=1.0)
        masks = torch.nan_to_num(masks, nan=0.0, posinf=1.0, neginf=0.0)
        
        # === 方法1：软分配方差（原方法，保留） ===
        slot_counts_soft = masks.sum(dim=2)  # (B, S) - 每个slot的软分配点数
        slot_counts_soft = torch.clamp(slot_counts_soft, min=0.0, max=float(N))
        
        slot_counts_mean_soft = slot_counts_soft.mean(dim=1, keepdim=True)  # (B, 1)
        variance_soft = ((slot_counts_soft - slot_counts_mean_soft) ** 2).mean(dim=1)  # (B,)
        
        ideal_count = N / S
        normalized_variance_soft = variance_soft / (ideal_count ** 2 + 1e-8)
        normalized_variance_soft = torch.clamp(normalized_variance_soft, min=0.0, max=100.0)
        
        # === 方法2：硬分配方差（新增，基于argmax） ===
        # 计算每个点分配到哪个slot（硬分配）
        hard_assignments = torch.argmax(masks, dim=1)  # (B, N) - 每个点的slot索引
        
        # 统计每个slot分配到的点数
        slot_counts_hard = torch.zeros(B, S, device=masks.device)
        for b in range(B):
            slot_counts_hard[b] = torch.bincount(
                hard_assignments[b], 
                minlength=S
            ).float()
        
        slot_counts_mean_hard = slot_counts_hard.mean(dim=1, keepdim=True)  # (B, 1)
        variance_hard = ((slot_counts_hard - slot_counts_mean_hard) ** 2).mean(dim=1)  # (B,)
        
        normalized_variance_hard = variance_hard / (ideal_count ** 2 + 1e-8)
        normalized_variance_hard = torch.clamp(normalized_variance_hard, min=0.0, max=100.0)
        
        # === 综合两种方法（硬分配权重更大） ===
        # 软分配：20%权重，硬分配：80%权重（增加硬分配权重）
        combined_variance = 0.2 * normalized_variance_soft + 0.8 * normalized_variance_hard
        
        return combined_variance.mean()
    
    def forward(self, reconstruction, sp_feats_proj, slots, masks):
        """
        计算总损失
        
        Args:
            reconstruction: (B, N, D)
            sp_feats_proj: (B, N, D) - SPFormer投影后的特征（目标）
            slots: (B, S, D_slot)
            masks: (B, S, N)
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 计算各项损失
        loss_recon = self.reconstruction_loss(reconstruction, sp_feats_proj)
        loss_entropy = self.mask_entropy_loss(masks)
        loss_diversity = self.slot_diversity_loss(slots)
        loss_uniformity = self.mask_uniformity_loss(masks)  # 新增
        
        # 加权求和
        total_loss = (
            self.w_recon * loss_recon +
            self.w_entropy * loss_entropy +
            self.w_diversity * loss_diversity +
            self.w_uniformity * loss_uniformity  # 新增
        )
        
        # 返回损失字典（用于记录）
        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': loss_recon.item(),
            'mask_entropy': loss_entropy.item(),
            'slot_diversity': loss_diversity.item(),
            'mask_uniformity': loss_uniformity.item()  # 新增
        }
        
        return total_loss, loss_dict

