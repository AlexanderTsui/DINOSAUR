"""
DINOSAUR 3D训练损失函数
包含4项损失:
1. Reconstruction Loss (MSE)
2. Mask Entropy Loss
3. Slot Diversity Loss
4. Mask Sparsity Loss
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
                - mask_sparsity
        """
        super().__init__()
        
        self.w_recon = loss_weights.get('reconstruction', 1.0)
        self.w_entropy = loss_weights.get('mask_entropy', 0.15)
        self.w_diversity = loss_weights.get('slot_diversity', 0.08)
        self.w_sparsity = loss_weights.get('mask_sparsity', 0.05)
        
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
        Slot多样性损失 - 防止Slot坍塌到相同表示
        
        Args:
            slots: (B, S, D_slot)
        
        计算Slot之间的余弦相似度，惩罚过高的相似度
        """
        B, S, D = slots.shape
        
        # 检测空slot（L2范数接近0）
        slot_norms = torch.norm(slots, p=2, dim=-1)  # (B, S)
        valid_slots = slot_norms > 1e-6
        
        # 至少需要2个有效slot才能计算diversity
        if valid_slots.sum(dim=1).min() < 2:
            return torch.tensor(0.0, device=slots.device)
        
        # L2归一化（添加eps防止除0）
        slots_norm = F.normalize(slots, p=2, dim=-1, eps=1e-8)  # (B, S, D)
        
        # 计算相似度矩阵
        similarity = torch.bmm(slots_norm, slots_norm.transpose(1, 2))  # (B, S, S)
        
        # 去除对角线（自己和自己的相似度）
        mask = ~torch.eye(S, dtype=torch.bool, device=slots.device).unsqueeze(0).expand(B, S, S)
        similarity_off_diag = similarity[mask].reshape(B, S, S-1)
        
        # 只对有效slot对计算损失
        # 为简化，这里计算所有非对角元素的平均（包含一些0 slot的影响）
        # 更精确的做法是只计算valid slot之间的相似度，但会更复杂
        
        # 惩罚高相似度（希望slot彼此不同）
        return similarity_off_diag.abs().mean()
    
    def mask_sparsity_loss(self, masks):
        """
        Mask稀疏性损失 - 鼓励稀疏分配
        
        Args:
            masks: (B, S, N)
        
        使用L1范数鼓励大部分mask值接近0或1
        注：空slot的mask全为0也是合理的，不需要特殊处理
        """
        return masks.abs().mean()
    
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
        loss_sparsity = self.mask_sparsity_loss(masks)
        
        # 加权求和
        total_loss = (
            self.w_recon * loss_recon +
            self.w_entropy * loss_entropy +
            self.w_diversity * loss_diversity +
            self.w_sparsity * loss_sparsity
        )
        
        # 返回损失字典（用于记录）
        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': loss_recon.item(),
            'mask_entropy': loss_entropy.item(),
            'slot_diversity': loss_diversity.item(),
            'mask_sparsity': loss_sparsity.item()
        }
        
        return total_loss, loss_dict

