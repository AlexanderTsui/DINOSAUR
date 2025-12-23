"""
DINOSAUR 3D训练损失函数
当前默认使用的无监督分解/聚类式损失（点级语义特征为目标）:
1. Feature Reconstruction Loss:  L_feat_rec
2. Intra-slot Compactness Loss:  L_compact
3. Assignment Entropy Loss:      L_entropy
4. Min-usage Loss (anti-collapse, long-tail friendly): L_min_usage
5. Optional Smoothness Loss (contrast-sensitive on kNN graph): L_smooth
6. Optional Consistency Loss (two-view): L_cons
7. Contrastive Learning Losses (NEW):
   - Intra-slot Compactness (prototype-based)
   - Inter-slot Separation (diversity)
   - Foreground-Background Contrast (Two-Stage specific)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ContrastiveLoss(nn.Module):
    """
    对比学习损失：增强slot对物体级特征的绑定能力

    包含三个子损失：
    1. Intra-slot Compactness: slot内特征应该相似
    2. Inter-slot Separation: 不同slot的特征应该不同
    3. Foreground-Background Contrast: 前景slot远离背景特征（Two-Stage专用）
    """

    def __init__(self, temperature=0.07, bg_slot_index=0):
        super().__init__()
        self.temperature = temperature
        self.bg_slot_index = bg_slot_index
        self.eps = 1e-8

    def compute_slot_prototypes(self, features, masks):
        """
        计算每个slot的原型特征（加权平均）

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
        mask_sums = masks.sum(dim=2, keepdim=True).clamp(min=self.eps)  # (B, S, 1)
        prototypes = prototypes / mask_sums  # (B, S, D)

        return F.normalize(prototypes, dim=-1, eps=self.eps)  # L2归一化

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
        features_norm = F.normalize(features, dim=-1, eps=self.eps)  # (B, N, D)

        # 计算每个point到所属slot prototype的相似度
        features_exp = features_norm.unsqueeze(1)  # (B, 1, N, D)
        prototypes_exp = prototypes.unsqueeze(2)  # (B, S, 1, D)

        # 余弦相似度
        similarities = (features_exp * prototypes_exp).sum(dim=-1)  # (B, S, N)

        # 加权损失：mask越大的点权重越高
        weighted_sim = (masks * similarities).sum(dim=(1, 2))  # (B,)
        mask_sum = masks.sum(dim=(1, 2)).clamp(min=self.eps)  # (B,)

        # 损失：最大化相似度 = 最小化 -similarity
        loss = -(weighted_sim / mask_sum).mean()

        return loss

    def inter_slot_separation(self, prototypes, exclude_bg=True):
        """
        Slot间分离性：鼓励不同slots的prototypes尽量不同

        Args:
            prototypes: (B, S, D)
            exclude_bg: 是否排除背景slot
        Returns:
            loss: scalar
        """
        B, S, D = prototypes.shape

        if exclude_bg and S > 1:
            # 只计算前景slots（slot 1~S-1）之间的相似度
            fg_prototypes = prototypes[:, 1:, :]  # (B, S-1, D)
        else:
            fg_prototypes = prototypes

        S_fg = fg_prototypes.shape[1]
        if S_fg <= 1:
            return torch.tensor(0.0, device=prototypes.device)

        # 计算pairwise余弦相似度矩阵
        # (B, S', D) x (B, D, S') -> (B, S', S')
        sim_matrix = torch.bmm(fg_prototypes, fg_prototypes.transpose(1, 2))

        # 只取上三角（避免重复计算和对角线）
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
            return torch.tensor(0.0, device=features.device)

        # 背景prototype
        bg_prototype = prototypes[:, self.bg_slot_index, :]  # (B, D)

        # 前景prototypes
        if self.bg_slot_index == 0:
            fg_prototypes = prototypes[:, 1:, :]  # (B, S-1, D)
        else:
            fg_prototypes = torch.cat([
                prototypes[:, :self.bg_slot_index, :],
                prototypes[:, self.bg_slot_index+1:, :]
            ], dim=1)  # (B, S-1, D)

        # 计算前景prototypes与背景prototype的相似度
        bg_proto_exp = bg_prototype.unsqueeze(1)  # (B, 1, D)
        similarities = (fg_prototypes * bg_proto_exp).sum(dim=-1)  # (B, S-1)

        # 损失：最小化前景-背景相似度（鼓励分离），margin=0.1
        loss = F.relu(similarities + 0.1).mean()

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
            loss_fg_bg = torch.tensor(0.0, device=features.device)

        return {
            'compact': loss_compact,
            'separate': loss_separate,
            'fg_bg_contrast': loss_fg_bg
        }


class DINOSAURLoss(nn.Module):
    """
    DINOSAUR损失函数
    """
    
    def __init__(self, loss_cfg):
        """
        Args:
            loss_cfg: dict，建议来自 config['loss']，形如:
              loss:
                weights:
                  feat_rec: 1.0
                  compact: 1.0
                  entropy: 0.0
                  min_usage: 0.0
                  smooth: 0.0
                  cons: 0.0
                params:
                  stop_grad_target: True
                  stop_grad_compact: True
                  min_usage_rho: 0.01
                  smooth_enabled: False
                  smooth_k: 16
                  smooth_sigma_x: 0.10
                  smooth_sigma_f: 0.50
                  smooth_use_feature_weight: True
                  smooth_use_entropy_gating: True
                  smooth_entropy_gating_power: 1.0
                  eps: 1e-8

            兼容旧版：如果直接传 dict 且不含 weights，则认为该 dict 就是 weights。
        """
        super().__init__()

        if isinstance(loss_cfg, dict) and 'weights' in loss_cfg:
            weights = loss_cfg.get('weights', {}) or {}
            params = loss_cfg.get('params', {}) or {}
        else:
            weights = loss_cfg or {}
            params = {}

        # 权重（兼容旧 key）
        self.w_feat_rec = float(weights.get('feat_rec', weights.get('reconstruction', 1.0)))
        self.w_compact = float(weights.get('compact', 0.0))
        self.w_entropy = float(weights.get('entropy', weights.get('mask_entropy', 0.0)))
        self.w_min_usage = float(weights.get('min_usage', 0.0))
        self.w_smooth = float(weights.get('smooth', 0.0))
        self.w_cons = float(weights.get('cons', 0.0))
        self.w_diversity = float(weights.get('diversity', weights.get('slot_diversity', 0.0)))
        # (D) 背景面积/点数偏置：鼓励 slot0 覆盖更多 token（谨慎开启，默认 0）
        self.w_bg_area = float(weights.get('bg_area', weights.get('background_area', 0.0)))

        # 对比学习损失权重（NEW）
        self.w_contrastive_compact = float(weights.get('contrastive_compact', 0.0))
        self.w_contrastive_separate = float(weights.get('contrastive_separate', 0.0))
        self.w_contrastive_fg_bg = float(weights.get('contrastive_fg_bg', 0.0))

        # 参数
        self.stop_grad_target = bool(params.get('stop_grad_target', True))
        self.stop_grad_compact = bool(params.get('stop_grad_compact', True))
        self.min_usage_rho = float(params.get('min_usage_rho', 0.01))

        self.smooth_enabled = bool(params.get('smooth_enabled', False))
        self.smooth_k = int(params.get('smooth_k', 16))
        self.smooth_sigma_x = float(params.get('smooth_sigma_x', 0.10))
        self.smooth_sigma_f = float(params.get('smooth_sigma_f', 0.50))
        self.smooth_use_feature_weight = bool(params.get('smooth_use_feature_weight', True))
        self.smooth_use_entropy_gating = bool(params.get('smooth_use_entropy_gating', True))
        self.smooth_entropy_gating_power = float(params.get('smooth_entropy_gating_power', 1.0))

        # entropy 正则：点级低熵(更硬) + 可选slot负载均衡(防塌缩到单slot)
        # - entropy_balance_weight = 0: 保持旧行为（仅点级低熵）
        # - >0: 增加 KL( mean_n p(slot|point) || Uniform )，鼓励slot使用率更均匀
        self.entropy_balance_weight = float(params.get('entropy_balance_weight', 0.0))

        # compact（slot内紧致）对离群点的敏感度控制：
        # 使用广义均值 (Generalized mean) 聚合每点到slot中心方向的余弦距离 dist∈[0,1]：
        #   L_k = ( E_w[ dist^p ] )^(1/p)
        # - p=1: 退化为加权平均（不敏感）
        # - p>1: 越大越“近似max”，对离群点更敏感
        # 另外可用 sharpen_gamma>1 强化高置信分配点（w^gamma）。
        self.compact_outlier_p = float(params.get('compact_outlier_p', 1.0))
        self.compact_sharpen_gamma = float(params.get('compact_sharpen_gamma', 1.0))

        # slot 多样性（鼓励不同 slot 学到不同表征）：
        # 在 slots 上做两两余弦相似度约束（只惩罚“过于相似”的正相关）。
        #   loss = mean_{i!=j} relu(cos(sloti, slotj) - margin)^power
        # - margin: 允许一定相似度（建议 0~0.2）
        # - power: 2 更平滑；1 更线性
        self.slot_diversity_margin = float(params.get('slot_diversity_margin', 0.0))
        self.slot_diversity_power = float(params.get('slot_diversity_power', 2.0))

        # (D) 背景面积正则参数
        # - bg_slot_index: 哪个 slot 视为背景（默认 0）
        # - bg_area_target: 若给定(0~1)，则使 mean(mask_bg) 接近该目标；否则用 -mean(mask_bg) 形式最大化背景覆盖
        self.bg_slot_index = int(params.get('bg_slot_index', 0))
        self.bg_area_target = params.get('bg_area_target', None)

        self.eps = float(params.get('eps', 1e-8))

        # 初始化对比学习损失模块（NEW）
        if self.w_contrastive_compact > 0 or self.w_contrastive_separate > 0 or self.w_contrastive_fg_bg > 0:
            self.contrastive_loss = ContrastiveLoss(
                temperature=params.get('contrastive_temperature', 0.07),
                bg_slot_index=params.get('bg_slot_index', 0)
            )
        else:
            self.contrastive_loss = None

        self.mse = nn.MSELoss()
    
    def feat_reconstruction_loss(self, reconstruction, target):
        """
        特征重建损失 (MSE)，目标特征默认 stop-grad
        
        Args:
            reconstruction: (B, N, D)
            target: (B, N, D)
        """
        tgt = target.detach() if self.stop_grad_target else target
        return self.mse(reconstruction, tgt)
    
    def entropy_loss(self, masks):
        """
        分配熵正则：
        1) 点级低熵（鼓励更硬分配）
        2) 可选：slot负载均衡（防止所有点塌缩到同一个slot）
        
        Args:
            masks: (B, S, N) - 每个超点在S个Slot上的分布
        
        返回一个标量 loss，越小越好
        """
        # masks应该已经在[0,1]且sum(dim=1)=1
        # 为安全起见，重新归一化（防止空slot导致的数值问题）
        masks_sum = masks.sum(dim=1, keepdim=True)
        masks = masks / (masks_sum + 1e-8)
        
        # 过滤掉全为0的mask（空slot或无效点）
        valid_mask = (masks_sum.squeeze(1) > 1e-6)  # (B, N)
        
        # 1) 点级熵: H = -sum_s p_s log p_s
        entropy_per_point = -(masks * torch.log(masks + 1e-8)).sum(dim=1)  # (B, N)
        
        # 只计算有效点的熵
        if valid_mask.sum() > 0:
            point_entropy = entropy_per_point[valid_mask].mean()
        else:
            point_entropy = torch.tensor(0.0, device=masks.device)

        # 2) slot 负载均衡：q_k = mean_n p(k|n)，最小化 KL(q || Uniform)
        # 这项专门用来抑制“所有点都去同一个slot”的全局塌缩解
        if self.entropy_balance_weight > 0:
            B, S, N = masks.shape
            q = masks.mean(dim=2)  # (B,S)
            q = q / (q.sum(dim=1, keepdim=True) + self.eps)
            q = torch.clamp(q, min=self.eps, max=1.0)
            # KL(q || U) = sum_k q_k (log q_k - log(1/S))
            log_uniform = -math.log(float(S))
            kl = (q * (torch.log(q) - log_uniform)).sum(dim=1).mean()
            return point_entropy + self.entropy_balance_weight * kl

        return point_entropy

    def compactness_loss(self, masks, feats):
        """
        Intra-slot compactness（类内紧致/一致性）:
        方案A：slot 内 pairwise 余弦紧致性（不显式计算 NxN 两两相似度，数值更稳）。

        目标：同一个 slot 内（按 soft mask 加权采样）的点特征两两更相似：
          L = 1 - E_{i,j~w_k}[ <f_i, f_j> ] ，其中 f 已做 L2 normalize。

        利用恒等式（避免 O(N^2)）：
          令 m_k = (sum_i w_{k,i} * f_i) / (sum_i w_{k,i})
          则 E_{i,j}[ <f_i, f_j> ] = ||m_k||^2
          所以 L_k = 1 - ||m_k||^2 ，天然在 [0, 1]，更便于调权重。
        
        Args:
            masks: (B, S, N)
            feats: (B, N, D)
        """

        B, S, N = masks.shape
        feats_in = feats.detach() if self.stop_grad_compact else feats

        # 归一化 masks，避免数值问题
        masks_sum = masks.sum(dim=1, keepdim=True)  # (B,1,N)
        masks = masks / (masks_sum + self.eps)
        masks = torch.clamp(masks, min=0.0, max=1.0)

        # slot-wise weights（可选 sharpen，强化高置信点；对离群点“高置信误分”会更敏感）
        w = masks  # (B,S,N)
        gamma = float(self.compact_sharpen_gamma)
        if gamma != 1.0:
            # 避免 0^gamma 的数值问题
            w = torch.clamp(w, min=0.0).pow(gamma)

        denom = w.sum(dim=2, keepdim=True)  # (B,S,1)

        # normalize feats for cosine geometry (stable grads with eps)
        eps_norm = max(float(self.eps), 1e-6)
        feats_n = F.normalize(feats_in, dim=-1, eps=eps_norm)  # (B,N,D)

        # v_k = sum_i w_{k,i} * f_i  -> (B,S,D)
        v = torch.bmm(w, feats_n)
        m = v / (denom + self.eps)  # (B,S,D)  (not unit)

        # slot中心方向（单位向量）
        m_dir = F.normalize(m, dim=-1, eps=eps_norm)  # (B,S,D)

        # per-point cosine distance to slot center direction in [0,1]
        cos = (feats_n.unsqueeze(1) * m_dir.unsqueeze(2)).sum(dim=-1).clamp(-1.0, 1.0)  # (B,S,N)
        dist = (1.0 - cos) * 0.5  # (B,S,N) in [0,1]

        # 广义均值聚合：对离群点敏感（p越大越像max）
        p = float(self.compact_outlier_p)
        p = 1.0 if p < 1.0 else p
        dist_p = dist.pow(p)
        mean_dist_p = (w * dist_p).sum(dim=2) / (denom.squeeze(-1) + self.eps)  # (B,S)
        loss_per_slot = mean_dist_p.clamp(0.0, 1.0).pow(1.0 / p)  # (B,S) in [0,1]

        # 对“极小占用”的 slot 不施加 compact（避免空slot被强行拉扯；占用由 min_usage/entropy 去调）
        denom_s = denom.squeeze(-1)  # (B,S)
        min_occ = max(1.0, 0.001 * float(N))  # soft count threshold
        valid = denom_s > min_occ
        if valid.any():
            return loss_per_slot[valid].mean()
        else:
            return torch.tensor(0.0, device=masks.device)

    def slot_diversity_loss(self, slots):
        """
        Slot 多样性损失：鼓励不同 slot 的表示彼此区分。

        Args:
            slots: (B, S, D_slot)
        Returns:
            scalar, 越小越好
        """
        if slots is None:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
        if slots.dim() != 3:
            return torch.tensor(0.0, device=slots.device)
        B, S, D = slots.shape
        if S <= 1:
            return torch.tensor(0.0, device=slots.device)

        eps_norm = max(float(self.eps), 1e-6)
        x = F.normalize(slots, dim=-1, eps=eps_norm)  # (B,S,D)
        sim = torch.bmm(x, x.transpose(1, 2)).clamp(-1.0, 1.0)  # (B,S,S)

        # off-diagonal mask
        eye = torch.eye(S, device=slots.device, dtype=slots.dtype).unsqueeze(0)  # (1,S,S)
        off = (1.0 - eye)

        margin = float(self.slot_diversity_margin)
        power = float(self.slot_diversity_power)
        power = 1.0 if power < 1.0 else power

        # only penalize overly positive similarity
        penalty = F.relu(sim - margin).pow(power) * off
        denom = off.sum() * float(B) + self.eps
        return penalty.sum() / denom

    def min_usage_loss(self, masks):
        """
        Min-usage anti-collapse（长尾友好）:
        只惩罚“占用率太小/空”的 slot，不强制均匀分割。
        
        Args:
            masks: (B, S, N) - 每个超点在S个Slot上的分布
        """
        B, S, N = masks.shape
        # 软占用率（比例）
        occ = masks.sum(dim=2) / (float(N) + self.eps)  # (B,S)
        # 只惩罚 occ < rho
        rho = self.min_usage_rho
        return F.relu(rho - occ).pow(2).mean()

    def background_area_loss(self, masks: torch.Tensor) -> torch.Tensor:
        """
        (D) 背景面积/点数正则：
        - 若设置 bg_area_target:  (mean_bg - target)^2
        - 否则:                 -mean_bg  （最大化背景覆盖）
        """
        if self.w_bg_area <= 0:
            return torch.tensor(0.0, device=masks.device)
        if masks is None or masks.dim() != 3:
            return torch.tensor(0.0, device=masks.device)
        B, S, N = masks.shape
        k = int(self.bg_slot_index)
        if k < 0 or k >= S:
            return torch.tensor(0.0, device=masks.device)
        bg = masks[:, k, :]  # (B,N)
        mean_bg = bg.mean()
        if self.bg_area_target is None:
            return -mean_bg
        tgt = float(self.bg_area_target)
        return (mean_bg - tgt) ** 2

    def smoothness_loss(self, masks, coords, feats=None):
        """
        Contrast-sensitive smoothness on kNN graph (optional).

        Args:
            masks: (B,S,N)
            coords: (B,N,3)
            feats: (B,N,D) optional，用于特征相似度加权
        """
        if (not self.smooth_enabled) or self.w_smooth <= 0:
            return torch.tensor(0.0, device=masks.device)
        if coords is None:
            return torch.tensor(0.0, device=masks.device)

        # 依赖 scipy 的 KDTree（CPU），在训练里可通过 w_smooth=0 关闭
        try:
            from scipy.spatial import cKDTree
        except Exception:
            return torch.tensor(0.0, device=masks.device)

        B, S, N = masks.shape
        k = max(1, int(self.smooth_k))

        # (B,N,S)
        A = masks.transpose(1, 2)
        A = A / (A.sum(dim=-1, keepdim=True) + self.eps)
        A = torch.clamp(A, min=0.0, max=1.0)

        # 预计算点级不确定度（用于 gating）
        if self.smooth_use_entropy_gating:
            ent = -(A * torch.log(A + self.eps)).sum(dim=-1)  # (B,N)
            ent = ent / (float(torch.log(torch.tensor(float(S), device=A.device))) + self.eps)
            ent = torch.clamp(ent, 0.0, 1.0)
        else:
            ent = None

        if feats is not None and self.smooth_use_feature_weight:
            f = feats.detach() if self.stop_grad_compact else feats
            f = F.normalize(f, dim=-1, eps=1e-6)  # (B,N,D)
        else:
            f = None

        total = 0.0
        denom = 0
        for b in range(B):
            coords_np = coords[b].detach().float().cpu().numpy()
            tree = cKDTree(coords_np)
            dist_x, nn_idx = tree.query(coords_np, k=k + 1)
            # drop self
            dist_x = torch.tensor(dist_x[:, 1:], device=masks.device, dtype=torch.float32)  # (N,k)
            nn_idx = torch.tensor(nn_idx[:, 1:], device=masks.device, dtype=torch.long)     # (N,k)

            Ab = A[b]  # (N,S)
            Ai = Ab.unsqueeze(1)           # (N,1,S)
            Aj = Ab[nn_idx]                # (N,k,S)
            diff2 = (Ai - Aj).pow(2).sum(dim=-1)  # (N,k)

            # weight from geometry
            wx = torch.exp(-(dist_x.pow(2)) / (self.smooth_sigma_x ** 2 + self.eps))

            w = wx
            # weight from feature similarity
            if f is not None:
                fb = f[b]  # (N,D)
                fi = fb.unsqueeze(1)       # (N,1,D)
                fj = fb[nn_idx]            # (N,k,D)
                cos = (fi * fj).sum(dim=-1).clamp(-1.0, 1.0)  # (N,k)
                df = (1.0 - cos)
                wf = torch.exp(-(df.pow(2)) / (self.smooth_sigma_f ** 2 + self.eps))
                w = w * wf

            # entropy gating: only smooth uncertain points
            if ent is not None:
                eb = ent[b]  # (N,)
                ei = eb.unsqueeze(1)       # (N,1)
                ej = eb[nn_idx]            # (N,k)
                g = torch.minimum(ei, ej).pow(self.smooth_entropy_gating_power)
                w = w * g

            loss_b = (w * diff2).mean()
            total = total + loss_b
            denom += 1

        return total / max(denom, 1)

    def consistency_loss(self, masks_view1, masks_view2):
        """
        Two-view consistency (optional). 需要两视角的对齐/同点对应；
        当前训练脚本未生成第二视角时，会自动为 0。
        """
        if masks_view2 is None:
            return torch.tensor(0.0, device=masks_view1.device)
        # (B,N,S)
        p1 = masks_view1.transpose(1, 2)
        p2 = masks_view2.transpose(1, 2)
        p1 = p1 / (p1.sum(dim=-1, keepdim=True) + self.eps)
        p2 = p2 / (p2.sum(dim=-1, keepdim=True) + self.eps)
        kl = (p1 * (torch.log(p1 + self.eps) - torch.log(p2 + self.eps))).sum(dim=-1)  # (B,N)
        return kl.mean()
    
    def forward(self, reconstruction, sp_feats_proj, slots, masks, coords=None, masks_view2=None, use_two_stage=False):
        """
        计算总损失

        Args:
            reconstruction: (B, N, D)
            sp_feats_proj: (B, N, D) - 目标特征（经过投影的编码器输出）
            slots: (B, S, D_slot)
            masks: (B, S, N)
            coords: (B, N, 3) - superpoints坐标（用于 smooth，可为空）
            masks_view2: (B, S, N) - 第二视角分配（用于 consistency，可为空）
            use_two_stage: bool - 是否使用Two-Stage DINOSAUR（影响对比损失）

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 计算各项损失
        loss_feat_rec = self.feat_reconstruction_loss(reconstruction, sp_feats_proj)
        loss_compact = self.compactness_loss(masks, sp_feats_proj)
        loss_entropy = self.entropy_loss(masks)
        loss_min_usage = self.min_usage_loss(masks)
        loss_smooth = self.smoothness_loss(masks, coords, feats=sp_feats_proj)
        loss_cons = self.consistency_loss(masks, masks_view2)
        loss_diversity = self.slot_diversity_loss(slots)
        loss_bg_area = self.background_area_loss(masks)

        # 对比学习损失（NEW）
        loss_contrastive_compact = torch.tensor(0.0, device=masks.device)
        loss_contrastive_separate = torch.tensor(0.0, device=masks.device)
        loss_contrastive_fg_bg = torch.tensor(0.0, device=masks.device)

        if self.contrastive_loss is not None:
            # 对特征使用与compactness一致的stop_grad策略
            feats_for_contrastive = sp_feats_proj.detach() if self.stop_grad_compact else sp_feats_proj

            contrastive_losses = self.contrastive_loss(
                feats_for_contrastive,
                masks,
                use_two_stage=use_two_stage
            )

            loss_contrastive_compact = contrastive_losses['compact']
            loss_contrastive_separate = contrastive_losses['separate']
            loss_contrastive_fg_bg = contrastive_losses['fg_bg_contrast']

        total_loss = (
            self.w_feat_rec * loss_feat_rec +
            self.w_compact * loss_compact +
            self.w_entropy * loss_entropy +
            self.w_min_usage * loss_min_usage +
            self.w_smooth * loss_smooth +
            self.w_cons * loss_cons +
            self.w_diversity * loss_diversity +
            self.w_bg_area * loss_bg_area +
            self.w_contrastive_compact * loss_contrastive_compact +
            self.w_contrastive_separate * loss_contrastive_separate +
            self.w_contrastive_fg_bg * loss_contrastive_fg_bg
        )

        loss_dict = {
            'total': float(total_loss.detach().item()),
            'feat_rec': float(loss_feat_rec.detach().item()),
            'compact': float(loss_compact.detach().item()),
            'entropy': float(loss_entropy.detach().item()),
            'min_usage': float(loss_min_usage.detach().item()),
            'smooth': float(loss_smooth.detach().item()),
            'cons': float(loss_cons.detach().item()),
            'diversity': float(loss_diversity.detach().item()),
            'bg_area': float(loss_bg_area.detach().item()),
            'contrastive_compact': float(loss_contrastive_compact.detach().item()),
            'contrastive_separate': float(loss_contrastive_separate.detach().item()),
            'contrastive_fg_bg': float(loss_contrastive_fg_bg.detach().item()),
        }

        return total_loss, loss_dict

