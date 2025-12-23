import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as F
from torch.nn import init
import random
import timm
from types import SimpleNamespace
from typing import Optional

from sklearn.cluster import AgglomerativeClustering


class GeoPE3DPositionalEncoding(nn.Module):
    """
    GeoPE (Geometric Positional Embedding) 的 3D 版本（参考 arXiv:2512.04963）。

    核心思想：
    - 将相对位移 (dx, dy, dz) 映射为耦合的 3D 旋转（SO(3)），用四元数实现；
    - 用该旋转去“旋转”一组可学习的 basis 向量，从而得到 D 维位置嵌入；
    - 与传统 Fourier 特征相比，这里各轴是几何耦合的，且更贴近欧氏几何不变性。

    注意：
    - 为了适配任意 D_slot，本实现只对前 floor(D/3)*3 维按 3 维一组旋转，剩余维度保持不变。
    """

    def __init__(self, dim: int, base: float = 100.0, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.eps = float(eps)

        self.group_dim = (dim // 3) * 3
        self.num_groups = dim // 3

        # 可学习的“向量 basis”，通过 GeoPE 旋转得到位置嵌入
        self.basis = nn.Parameter(torch.randn(1, 1, 1, dim) * 0.02)

        # 频率：与 RoPE 类似的跨通道缩放（每个 3D 子向量一组）
        if self.num_groups > 0:
            i = torch.arange(self.num_groups).float()  # (G,)
            inv_freq = self.base ** (2 * i / dim)      # (G,)
            self.register_buffer("inv_freq", inv_freq)
        else:
            self.register_buffer("inv_freq", torch.empty(0))

    @staticmethod
    def _quat_rotate(v: torch.Tensor, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        旋转向量 v（... , 3），四元数 q = (s, u) 其中 s 标量，u 向量部分（... , 3）
        使用等价的向量形式：v' = v + 2*s*(u×v) + 2*(u×(u×v))
        """
        uv = torch.cross(u, v, dim=-1)
        uuv = torch.cross(u, uv, dim=-1)
        return v + 2.0 * s * uv + 2.0 * uuv

    def forward(self, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rel_pos: (..., 3)  3D 相对位移（建议已做适度归一化/裁剪）
        Returns:
            pe: (..., D)  GeoPE 位置嵌入
        """
        assert rel_pos.shape[-1] == 3

        *prefix, _ = rel_pos.shape
        pe = self.basis.expand(*prefix, self.dim)  # (..., D)

        if self.num_groups == 0:
            return pe

        # 仅对前 group_dim 维做 3D 旋转
        pe_main = pe[..., : self.group_dim].reshape(*prefix, self.num_groups, 3)  # (..., G, 3)

        # theta: (..., G, 3)  (对应论文里的 θ_d, θ_h, θ_w 的耦合相位向量)
        theta = rel_pos.unsqueeze(-2) * self.inv_freq.view(*([1] * len(prefix)), self.num_groups, 1)

        # Θ = (1/3) * ||theta||_2  （论文 3D 版本的 composite phase）
        Theta = (theta.pow(2).sum(dim=-1, keepdim=True).sqrt()) / 3.0  # (..., G, 1)

        half = 0.5 * Theta
        s = torch.cos(half)  # (..., G, 1)

        # u = sin(Θ/2) * theta / (3*Θ) ；当 Θ→0 时用 eps 保持稳定
        denom = 3.0 * Theta + self.eps
        u = torch.sin(half) * theta / denom  # (..., G, 3)

        pe_rot = self._quat_rotate(pe_main, s, u)  # (..., G, 3)
        pe_rot = pe_rot.reshape(*prefix, self.group_dim)  # (..., group_dim)

        if self.group_dim == self.dim:
            return pe_rot

        return torch.cat([pe_rot, pe[..., self.group_dim :]], dim=-1)


class Loss_Function(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.token_num = args.token_num
        self.num_slots = args.num_slots

        self.epsilon = 1e-8

    def forward(self, reconstruction, masks, target):
        # :args reconstruction: (B, token, 768)
        # :args masks: (B, S, token)
        # :args target: (B, token, 768)

        target = target.detach()
        loss = self.mse(reconstruction, target.detach()).mean()

        return loss

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, residual=False, layer_order="none"):
        super().__init__()
        self.residual = residual
        self.layer_order = layer_order
        if residual:
            assert input_dim == output_dim

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)

        if layer_order in ["pre", "post"]:
            self.norm = nn.LayerNorm(input_dim)
        else:
            assert layer_order == "none"

    def forward(self, x):
        input = x

        if self.layer_order == "pre":
            x = self.norm(x)

        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.dropout(x)

        if self.residual:
            x = x + input
        if self.layer_order == "post":
            x = self.norm(x)

        return x
    
class Visual_Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.resize_to = args.resize_to
        self.token_num = args.token_num
        self.encoder = args.encoder
        self.model = self.load_model(args)


    def load_model(self, args):
        assert args.resize_to[0] % args.patch_size == 0
        assert args.resize_to[1] % args.patch_size == 0
        
        if args.encoder == "dino-vitb-8":
            model = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
        elif args.encoder == "dino-vitb-16":
            model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
        elif args.encoder == "dinov2-vitb-14":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        elif args.encoder == "sup-vitb-16":
            model = timm.create_model("vit_base_patch16_224", pretrained=True, img_size=(args.resize_to[0], args.resize_to[1]))
        else:
            assert False

        for p in model.parameters():
            p.requires_grad = False

        # wget https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth
        # wget https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth
        # wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
        
        return model
    
    @torch.no_grad()
    def forward(self, frames):
        # :arg frames:  (B, 3, H, W)
        #
        # :return x:  (B, token, 768)

        B = frames.shape[0]

        self.model.eval()

        if self.encoder.startswith("dinov2-"):
            x = self.model.prepare_tokens_with_masks(frames)
        elif self.encoder.startswith("sup-"):
            x = self.model.patch_embed(frames)
            x = self.model._pos_embed(x)
        else:
            x = self.model.prepare_tokens(frames)


        for blk in self.model.blocks:
            x = blk(x)
        x = x[:, 1:]

        assert x.shape[0] == B
        assert x.shape[1] == self.token_num
        assert x.shape[2] == 768

        return x



class MLPDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        # === Token calculations ===
        slot_dim = args.slot_dim
        hidden_dim = 2048

        # === MLP Based Decoder ===
        self.layer1 = nn.Linear(slot_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, 768 + 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, slot_maps):
        # :arg slot_maps: (B * S, token, D_slot)

        slot_maps = self.relu(self.layer1(slot_maps))    # (B * S, token, D_hidden)
        slot_maps = self.relu(self.layer2(slot_maps))    # (B * S, token, D_hidden)
        slot_maps = self.relu(self.layer3(slot_maps))    # (B * S, token, D_hidden)

        slot_maps = self.layer4(slot_maps)               # (B * S, token, 768 + 1)

        return slot_maps


class TransformerDecoder(nn.Module):
    """
    Transformer-based decoder（可选）：
    对每个 slot 的 token 序列做 self-attention，再投影到 (768 + 1)。

    注意：self-attention 复杂度 O(N^2)，token_num 很大（如 4096）时会明显变慢/吃显存。
    """

    def __init__(self, args):
        super().__init__()
        d_model = args.slot_dim

        n_layers = int(getattr(args, "decoder_tf_layers", 2))
        n_heads = int(getattr(args, "decoder_tf_heads", 8))
        ff_dim = int(getattr(args, "decoder_tf_ff_dim", 4 * d_model))
        dropout = float(getattr(args, "decoder_tf_dropout", 0.1))

        assert d_model % n_heads == 0, "decoder_tf_heads 必须整除 slot_dim"

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 768 + 1)

    def forward(self, slot_maps):
        # :arg slot_maps: (B * S, token, D_slot)
        x = self.encoder(slot_maps)         # (B * S, token, D_slot)
        x = self.out_norm(x)
        x = self.out_proj(x)                # (B * S, token, 768 + 1)
        return x
    
class ISA(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()

        self.num_slots = args.num_slots
        self.scale = args.slot_dim ** -0.5
        self.iters = args.slot_att_iter
        self.slot_dim = args.slot_dim
        self.query_opt = args.query_opt

        # === 3D点云配置（替代2D网格配置）===
        self.coord_dim = 3  # 3D坐标维度（x, y, z）
        self.sigma = 5  # 归一化因子
        # 注意：不再需要res_h, res_w和预定义的abs_grid
        # abs_grid将在forward中动态接收点云坐标
        # === === ===

        # === 位置编码配置（已移除 Fourier，统一使用 GeoPE）===
        self.use_geo_pe = getattr(args, "use_geo_pe", True)
        geo_base = getattr(args, "geo_pe_base", 100.0)
        self.geo_pe = GeoPE3DPositionalEncoding(dim=self.slot_dim, base=geo_base)
        # === === ===

        # === 背景 slot 先验（A/B/C 开关）===
        # - bg_slot_index: 背景 slot 编号（默认 0）
        # - bg_slot_mean_init: 用 token 特征均值初始化该 slot（B）
        # - bg_slot_no_pe: 对该 slot 关闭 GeoPE 注入（C）
        self.bg_slot_index = int(getattr(args, "bg_slot_index", 0))
        self.bg_slot_mean_init = bool(getattr(args, "bg_slot_mean_init", False))
        self.bg_slot_mean_init_detach = bool(getattr(args, "bg_slot_mean_init_detach", True))
        self.bg_slot_no_pe = bool(getattr(args, "bg_slot_no_pe", False))
        # === === ===

        # === Slot related ===
        if self.query_opt:
            self.slots = nn.Parameter(torch.Tensor(1, self.num_slots, self.slot_dim))
            init.xavier_uniform_(self.slots)
        else:
            self.slots_mu = nn.Parameter(torch.randn(1, 1, self.slot_dim))
            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, self.slot_dim))
            init.xavier_uniform_(self.slots_mu)
            init.xavier_uniform_(self.slots_logsigma)

        # 3D slot参考坐标系参数（从2D改为3D）
        self.S_s = nn.Parameter(torch.Tensor(1, self.num_slots, 1, 3))  # (1, S, 1, 3) - 3D尺度
        self.S_p = nn.Parameter(torch.Tensor(1, self.num_slots, 1, 3))  # (1, S, 1, 3) - 3D位置

        # 增大S_s初始值,避免除以过小的数导致数值爆炸
        init.normal_(self.S_s, mean=0.5, std=0.1)  # 从mean=0增加到0.5
        init.normal_(self.S_p, mean=0., std=.02)
        # === === ===

        # === Slot Attention related ===
        self.Q = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.norm = nn.LayerNorm(self.slot_dim)
        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.mlp = MLP(self.slot_dim, 4*self.slot_dim, self.slot_dim,
                       residual=True, layer_order="pre")
        # === === ===

        # === Query & Key & Value ===
        self.K = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.V = nn.Linear(self.slot_dim, self.slot_dim, bias=False)

        self.f = nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.slot_dim, self.slot_dim))
        # === === ===

        # Note: starts and ends with LayerNorm
        self.initial_mlp = nn.Sequential(nn.LayerNorm(input_dim),
                                         nn.Linear(input_dim, input_dim),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(input_dim, self.slot_dim),
                                         nn.LayerNorm(self.slot_dim))

        self.final_layer = nn.Linear(self.slot_dim, self.slot_dim)

    def get_rel_grid(self, attn, abs_grid):
        # :arg attn: (B, S, N) - attention权重
        # :arg abs_grid: (B, S, N, 3) - 3D点云坐标
        #
        # :return rel_grid: (B, S, N, D_slot) - 相对坐标编码

        B, S, N = attn.shape
        attn = attn.unsqueeze(dim=2)                                            # (B, S, 1, N)
        
        # 计算slot中心位置（3D）- 加权平均
        S_p = torch.einsum('bsjd,bsij->bsd', abs_grid, attn)                    # (B, S, N, 3) x (B, S, 1, N) -> (B, S, 3)
        S_p = S_p.unsqueeze(dim=2)                                              # (B, S, 1, 3)

        # 计算slot尺度（3D）- 加权方差的平方根
        values_ss = torch.pow(abs_grid - S_p, 2)                                # (B, S, N, 3)
        S_s = torch.einsum('bsjd,bsij->bsd', values_ss, attn)                   # (B, S, N, 3) x (B, S, 1, N) -> (B, S, 3)
        S_s = torch.sqrt(S_s)                                                   # (B, S, 3)
        S_s = S_s.unsqueeze(dim=2)                                              # (B, S, 1, 3)

        # 归一化得到相对坐标（3D）- 添加数值稳定性保护
        S_s_safe = torch.clamp(S_s, min=0.2) + 0.1                              # 确保S_s不会太小
        rel_grid = (abs_grid - S_p) / (S_s_safe * self.sigma + 1e-6)            # (B, S, N, 3)
        rel_grid = torch.clamp(rel_grid, min=-3, max=3)                         # 防止极端值
        
        # GeoPE：直接输出 (B,S,N,D_slot) 的几何位置嵌入（用于 decoder 的 slot_maps += rel_grid）
        return self.geo_pe(rel_grid)


    def forward(self, inputs, point_coords, token_weights: Optional[torch.Tensor] = None):
        # :arg inputs:              (B, N, D) - 点云特征
        # :arg point_coords:        (B, N, 3) - 点云的3D坐标（已归一化到[-1, 1]）
        # :arg token_weights:       (B, N) or (B, 1, N) - 可选，token权重（例如前景权重）；为0的token不会参与slot更新
        #
        # :return slots:            (B, S, D_slot) - slot表示
        # :return attn:             (B, S, N) - attention权重

        B, N, D = inputs.shape
        S = self.num_slots
        D_slot = self.slot_dim
        epsilon = 1e-8

        # 初始化slots
        if self.query_opt:
            slots = self.slots.expand(B, S, D_slot)                     # (B, S, D_slot)
        else:
            mu = self.slots_mu.expand(B, S, D_slot)
            sigma = self.slots_logsigma.exp().expand(B, S, D_slot)
            slots = mu + sigma * torch.randn(mu.shape, device=sigma.device, dtype=sigma.dtype)

        slots_init = slots
        
        # 预处理输入特征
        token_emb = self.initial_mlp(inputs)                        # (B, N, D_slot)
        # 检查并修复initial_mlp输出的NaN
        token_emb = torch.nan_to_num(token_emb, nan=0.0, posinf=1.0, neginf=-1.0)

        # (B) 背景 slot 均值初始化（slot space）
        if self.bg_slot_mean_init and (0 <= self.bg_slot_index < S):
            mean_init = token_emb.mean(dim=1)  # (B, D_slot)
            if self.bg_slot_mean_init_detach:
                mean_init = mean_init.detach()
            slots = slots.clone()
            slots[:, self.bg_slot_index, :] = mean_init

        inputs = token_emb.unsqueeze(dim=1)                         # (B, 1, N, D_slot)
        inputs = inputs.expand(B, S, N, D_slot)                     # (B, S, N, D_slot)

        # 构建3D abs_grid：从输入的点云坐标
        abs_grid = point_coords.unsqueeze(1)                        # (B, 1, N, 3)
        abs_grid = abs_grid.expand(B, S, N, 3)                      # (B, S, N, 3)

        assert torch.sum(torch.isnan(abs_grid)) == 0, "abs_grid包含NaN"
        assert torch.sum(torch.isnan(inputs)) == 0, "inputs包含NaN"

        # 初始化slot参考坐标系（3D）
        S_s = self.S_s.expand(B, S, 1, 3)                           # (B, S, 1, 3)
        S_p = self.S_p.expand(B, S, 1, 3)                           # (B, S, 1, 3)

        for t in range(self.iters + 1):
            # last iteration for S_s and S_p: t = self.iters
            # last meaningful iteration: t = self.iters - 1

            assert torch.sum(torch.isnan(slots)) == 0, f"Iteration {t}: slots包含NaN"
            assert torch.sum(torch.isnan(S_s)) == 0, f"Iteration {t}: S_s包含NaN"
            assert torch.sum(torch.isnan(S_p)) == 0, f"Iteration {t}: S_p包含NaN"
            
            if self.query_opt and (t == self.iters - 1):
                slots = slots.detach() + slots_init - slots_init.detach()

            slots_prev = slots
            slots = self.norm(slots)

            # === key and value calculation using rel_grid (3D) ===
            # 添加数值稳定性保护
            # 确保S_s不会过小（增大下限+偏移，防止训练中后期S_s收敛到0导致梯度爆炸）
            S_s_safe = torch.clamp(S_s, min=0.3, max=5.0) + 0.1      # 更保守的范围+偏移
            rel_grid = (abs_grid - S_p) / (S_s_safe * self.sigma + 1e-4)  # (B, S, N, 3) - 3D相对坐标
            rel_grid = torch.clamp(rel_grid, min=-3, max=3)          # 更严格的裁剪防止爆炸
            
            # 位置编码注入到 K/V：GeoPE 生成 (B,S,N,D_slot) 的位置嵌入，做加性注入
            pos = self.geo_pe(rel_grid)                                # (B, S, N, D_slot)
            # (C) 背景 slot 不注入位置编码（Residual/garbage slot）
            if self.bg_slot_no_pe and (0 <= self.bg_slot_index < S):
                pos = pos.clone()
                pos[:, self.bg_slot_index, :, :] = 0.0
            k = self.f(self.K(inputs) + pos)
            v = self.f(self.V(inputs) + pos)

            # === Calculate attention ===
            q = self.Q(slots).unsqueeze(dim=-1)                      # (B, S, D_slot, 1)

            dots = torch.einsum('bsdi,bsjd->bsj', q, k)              # (B, S, D_slot, 1) x (B, S, N, D_slot) -> (B, S, N)
            dots = dots * self.scale                                 # (B, S, N)
            # 更严格的裁剪防止softmax溢出（训练中后期dots可能变大）
            dots = torch.clamp(dots, min=-30, max=30)
            attn = dots.softmax(dim=1) + epsilon                     # (B, S, N) - softmax over slots

            # === Token weights (例如：只用前景token更新slot) ===
            if token_weights is not None:
                # 支持 (B,N) 或 (B,1,N)
                if token_weights.dim() == 2:
                    w = token_weights.unsqueeze(1)  # (B,1,N)
                elif token_weights.dim() == 3:
                    w = token_weights
                else:
                    raise ValueError(f"token_weights 维度非法: {token_weights.shape}")
                # clamp 避免出现负数/NaN
                w = torch.nan_to_num(w, nan=0.0, posinf=1.0, neginf=0.0)
                w = torch.clamp(w, min=0.0, max=1.0)
                attn = attn * w  # (B,S,N) 加权后，权重为0的token将不参与后续更新

            # === Weighted mean ===
            # 添加额外的epsilon防止空slot导致除0
            attn_sum = attn.sum(dim=-1, keepdim=True)                # (B, S, 1)
            attn = attn / (attn_sum + epsilon)                       # (B, S, N) - 归一化
            attn = attn.unsqueeze(dim=2)                             # (B, S, 1, N)
            updates = torch.einsum('bsjd,bsij->bsd', v, attn)        # (B, S, N, D_slot) x (B, S, 1, N) -> (B, S, D_slot)

            # === Update S_p and S_s (3D) ===
            S_p = torch.einsum('bsjd,bsij->bsd', abs_grid, attn)     # (B, S, N, 3) x (B, S, 1, N) -> (B, S, 3)
            S_p = S_p.unsqueeze(dim=2)                               # (B, S, 1, 3)

            values_ss = torch.pow(abs_grid - S_p, 2)                 # (B, S, N, 3)
            S_s = torch.einsum('bsjd,bsij->bsd', values_ss, attn)    # (B, S, N, 3) x (B, S, 1, N) -> (B, S, 3)
            S_s = torch.sqrt(S_s + 1e-6)                             # (B, S, 3) - 添加较大epsilon避免sqrt(0)梯度爆炸
            S_s = torch.clamp(S_s, min=0.2, max=5.0)                 # 更保守的范围，防止除0或爆炸
            S_s = S_s.unsqueeze(dim=2)                               # (B, S, 1, 3)

            # === Update slots (与坐标维度无关，保持不变) ===
            if t != self.iters:
                # 添加NaN检查和数值裁剪
                updates = torch.nan_to_num(updates, nan=0.0, posinf=1.0, neginf=-1.0)
                
                slots = self.gru(
                    updates.reshape(-1, self.slot_dim),
                    slots_prev.reshape(-1, self.slot_dim))

                slots = slots.reshape(B, -1, self.slot_dim)
                slots = self.mlp(slots)
                
                # 再次检查并修复NaN
                slots = torch.nan_to_num(slots, nan=0.0, posinf=1.0, neginf=-1.0)

        slots = self.final_layer(slots_prev)                         # (B, S, D_slot)
        attn = attn.squeeze(dim=2)                                   # (B, S, N)

        return slots, attn
    

class SA(nn.Module):
    def __init__(self, args, input_dim):
        
        super().__init__()
        self.num_slots = args.num_slots
        self.scale = args.slot_dim ** -0.5
        self.iters = args.slot_att_iter
        self.slot_dim = args.slot_dim
        self.query_opt = args.query_opt

        # === Slot related ===
        if self.query_opt:
            self.slots = nn.Parameter(torch.Tensor(1, self.num_slots, self.slot_dim))
            init.xavier_uniform_(self.slots)
        else:
            self.slots_mu = nn.Parameter(torch.randn(1, 1, self.slot_dim))
            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, self.slot_dim))
            init.xavier_uniform_(self.slots_mu)
            init.xavier_uniform_(self.slots_logsigma)

        # === Slot Attention related ===
        self.Q = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.norm = nn.LayerNorm(self.slot_dim)
        self.update_norm = nn.LayerNorm(self.slot_dim)
        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.mlp = MLP(self.slot_dim, 4 * self.slot_dim, self.slot_dim,
                       residual=True, layer_order="pre")
        # === === ===

        # === Query & Key & Value ===
        self.K = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.V = nn.Linear(self.slot_dim, self.slot_dim, bias=False)

        self.f = nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.slot_dim, self.slot_dim))
        # === === ===

        # Note: starts and ends with LayerNorm
        self.initial_mlp = nn.Sequential(nn.LayerNorm(input_dim),
                                         nn.Linear(input_dim, input_dim),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(input_dim, self.slot_dim),
                                         nn.LayerNorm(self.slot_dim))

        self.final_layer = nn.Linear(self.slot_dim, self.slot_dim)

        # (B) 背景 slot 均值初始化（SA 也可用；C 不适用于 SA）
        self.bg_slot_index = int(getattr(args, "bg_slot_index", 0))
        self.bg_slot_mean_init = bool(getattr(args, "bg_slot_mean_init", False))
        self.bg_slot_mean_init_detach = bool(getattr(args, "bg_slot_mean_init_detach", True))

    def forward(self, inputs, token_weights: Optional[torch.Tensor] = None):
        # :arg inputs:              (B, token, D)
        # :arg token_weights:       (B, token) or (B, 1, token) - 可选，token权重；为0的token不参与slot更新
        #
        # :return slots:            (B, S, D_slot)

        B = inputs.shape[0]
        S = self.num_slots
        D_slot = self.slot_dim
        epsilon = 1e-8

        if self.query_opt:
            slots = self.slots.expand(B, S, D_slot)          # (B, S, D_slot)
        else:
            mu = self.slots_mu.expand(B, S, D_slot)
            sigma = self.slots_logsigma.exp().expand(B, S, D_slot)
            slots = mu + sigma * torch.randn(mu.shape, device=sigma.device, dtype=sigma.dtype)

        slots_init = slots
        inputs = self.initial_mlp(inputs)                    # (B, token, D_slot)

        # (B) 背景 slot 均值初始化（slot space）
        if self.bg_slot_mean_init and (0 <= self.bg_slot_index < S):
            mean_init = inputs.mean(dim=1)  # (B, D_slot)
            if self.bg_slot_mean_init_detach:
                mean_init = mean_init.detach()
            slots = slots.clone()
            slots[:, self.bg_slot_index, :] = mean_init

        keys = self.K(inputs)                                # (B, token, D_slot)
        values = self.V(inputs)                              # (B, token, D_slot)
        
        for t in range(self.iters):
            assert torch.sum(torch.isnan(slots)) == 0, f"Iteration {t}"
            
            if t == self.iters - 1 and self.query_opt:
                slots = slots.detach() + slots_init - slots_init.detach()

            slots_prev = slots
            slots = self.norm(slots)
            queries = self.Q(slots)                                     # (B, S, D_slot)

            dots = torch.einsum('bsd,btd->bst', queries, keys)          # (B, S, token)
            dots *= self.scale                                          # (B, S, token)
            attn = dots.softmax(dim=1) + epsilon                        # (B, S, token)

            if token_weights is not None:
                if token_weights.dim() == 2:
                    w = token_weights.unsqueeze(1)  # (B,1,token)
                elif token_weights.dim() == 3:
                    w = token_weights
                else:
                    raise ValueError(f"token_weights 维度非法: {token_weights.shape}")
                w = torch.nan_to_num(w, nan=0.0, posinf=1.0, neginf=0.0)
                w = torch.clamp(w, min=0.0, max=1.0)
                attn = attn * w  # (B,S,token)

            attn = attn / attn.sum(dim=-1, keepdim=True)                # (B, S, token)

            updates = torch.einsum('bst,btd->bsd', attn, values)        # (B, S, D_slot)

            slots = self.gru(
                    updates.reshape(-1, self.slot_dim),
                    slots_prev.reshape(-1, self.slot_dim))

            slots = slots.reshape(B, -1, self.slot_dim)
            slots = self.mlp(slots)

        self.final_layer(slots)

        return slots




class DINOSAURpp(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.slot_dim = args.slot_dim
        self.slot_num = args.num_slots
        self.token_num = args.token_num

        self.ISA = args.ISA
        if self.ISA:
            self.slot_encoder = ISA(args, input_dim=768)
        else:
            self.slot_encoder = SA(args, input_dim=768)

        decoder_type = getattr(args, "decoder_type", "mlp")
        decoder_type = str(decoder_type).lower()
        if decoder_type in ["mlp", "ffn"]:
            self.slot_decoder = MLPDecoder(args)
        elif decoder_type in ["transformer", "tf", "transformer_decoder"]:
            self.slot_decoder = TransformerDecoder(args)
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

        self.pos_dec = nn.Parameter(torch.Tensor(1, self.token_num, self.slot_dim))
        init.normal_(self.pos_dec, mean=0., std=.02)

    def sbd_slots(self, slots):
        # :arg slots: (B, S, D_slot)
        # 
        # :return slots: (B, S, token, D_slot)

        B, S, D_slot = slots.shape

        slots = slots.view(-1, 1, D_slot)                   # (B * S, 1, D_slot)
        slots = slots.tile(1, self.token_num, 1)            # (B * S, token, D_slot)

        pos_embed = self.pos_dec.expand(slots.shape)
        slots = slots + pos_embed                          # (B * S, token, D_slot)
        slots = slots.view(B, S, self.token_num, D_slot)

        return slots
    
    
    def reconstruct_feature_map(self, slot_maps):
        # :arg slot_maps: (B, S, token, 768 + 1)
        #
        # :return reconstruction: (B, token, 768)
        # :return masks: (B, S, token)

        B = slot_maps.shape[0]

        channels, masks = torch.split(slot_maps, [768, 1], dim=-1)  # (B, S, token, 768), (B, S, token, 1)
        masks = masks.softmax(dim=1)                                # (B, S, token, 1)

        reconstruction = torch.sum(channels * masks, dim=1)         # (B, token, 768)
        masks = masks.squeeze(dim=-1)                               # (B, S, token)

        return reconstruction, masks


    def forward(self, features, point_coords=None):
        # :arg features: (B, N, feature_dim) - 点云特征
        # :arg point_coords: (B, N, 3) - 点云的3D坐标（已归一化），ISA模式下必需
        #
        # :return reconstruction: (B, N, feature_dim) - 重建特征
        # :return slots: (B, S, D_slot) - slot表示
        # :return masks: (B, S, N) - 分割mask

        B, N, _ = features.shape

        if self.ISA:
            # ISA模式：需要点云坐标
            assert point_coords is not None, "ISA模式需要提供point_coords参数"
            
            slots, attn = self.slot_encoder(features, point_coords)             # (B, S, D_slot), (B, S, N)
            assert torch.sum(torch.isnan(slots)) == 0, "slots包含NaN"
            assert torch.sum(torch.isnan(attn)) == 0, "attn包含NaN"

            # 构建abs_grid用于get_rel_grid
            abs_grid = point_coords.unsqueeze(1)                                # (B, 1, N, 3)
            abs_grid = abs_grid.expand(B, self.slot_num, N, 3)                  # (B, S, N, 3)
            
            rel_grid = self.slot_encoder.get_rel_grid(attn, abs_grid)           # (B, S, N, D_slot)

            slot_maps = self.sbd_slots(slots) + rel_grid                        # (B, S, N, D_slot)
            slot_maps = self.slot_decoder(slot_maps.reshape(B * self.slot_num, N, -1))  # (B*S, N, feature_dim+1)
            slot_maps = slot_maps.reshape(B, self.slot_num, N, -1)              # (B, S, N, feature_dim+1)
        
        else:
            # SA模式：不需要点云坐标
            slots = self.slot_encoder(features)                                 # (B, S, D_slot)
            assert torch.sum(torch.isnan(slots)) == 0, "slots包含NaN"

            slot_maps = self.sbd_slots(slots)                                   # (B, S, N, D_slot)
            slot_maps = self.slot_decoder(slot_maps.reshape(B * self.slot_num, N, -1))  # (B*S, N, feature_dim+1)
            slot_maps = slot_maps.reshape(B, self.slot_num, N, -1)              # (B, S, N, feature_dim+1)

        reconstruction, masks = self.reconstruct_feature_map(slot_maps)         # (B, N, feature_dim), (B, S, N)

        return reconstruction, slots, masks


def _args_with_overrides(args, **overrides):
    """
    将 Args(自定义类/Namespace) 转成 SimpleNamespace，并覆盖指定字段。
    目的：复用现有 args 的超参，快速构建 stage1/stage2 的独立配置。
    """
    base = {}
    # Args 既可能是自定义 class，也可能是 SimpleNamespace
    for k, v in vars(args).items():
        base[k] = v
    base.update(overrides)
    return SimpleNamespace(**base)


class TwoStageDINOSAURpp(nn.Module):
    """
    两阶段 Slot Attention（与现有训练接口兼容）：
      - Stage1: 2 个 slot（背景/前景），用于产生 fg/bg masks（来自 decoder 的 softmax masks）
      - Stage2: (S-1) 个 slot 仅在前景 token 上更新/竞争，输出前景细分 masks
      - Final: 输出总 slot 数仍为 S = args.num_slots：
          slot0 = background（来自 stage1）
          slot1..S-1 = foreground parts（来自 stage2）

    输出与 DINOSAURpp 保持一致：
      reconstruction: (B, N, 768)
      slots:          (B, S, D_slot)
      masks:          (B, S, N)  （按 slot 维度 softmax 后的概率）
    """

    def __init__(self, args):
        super().__init__()

        self.slot_dim = args.slot_dim
        self.slot_num = int(args.num_slots)
        self.token_num = int(args.token_num)
        self.ISA = bool(args.ISA)

        if self.slot_num < 2:
            raise ValueError(f"TwoStageDINOSAURpp 需要 num_slots>=2，但得到 {self.slot_num}")

        # stage1: 固定 2 slots（bg/fg）
        stage1_iters = int(getattr(args, "two_stage_stage1_iters", args.slot_att_iter))
        # (B/C) 默认仅对 stage1 开启：slot0 均值初始化 + slot0 不注入 GeoPE
        stage1_bg_mean_init = bool(getattr(args, "two_stage_stage1_bg_mean_init", True))
        stage1_bg_no_pe = bool(getattr(args, "two_stage_stage1_bg_no_pe", True))
        args_s1 = _args_with_overrides(
            args,
            num_slots=2,
            slot_att_iter=stage1_iters,
            bg_slot_index=0,
            bg_slot_mean_init=stage1_bg_mean_init,
            bg_slot_no_pe=stage1_bg_no_pe,
        )

        # stage2: (S-1) slots，用于前景细分
        stage2_iters = int(getattr(args, "two_stage_stage2_iters", args.slot_att_iter))
        # stage2 默认关闭 B/C（让前景 slots 更自由）
        args_s2 = _args_with_overrides(
            args,
            num_slots=self.slot_num - 1,
            slot_att_iter=stage2_iters,
            bg_slot_index=0,
            bg_slot_mean_init=False,
            bg_slot_no_pe=False,
        )

        # encoders
        if self.ISA:
            self.stage1_encoder = ISA(args_s1, input_dim=768)
            self.stage2_encoder = ISA(args_s2, input_dim=768)
        else:
            self.stage1_encoder = SA(args_s1, input_dim=768)
            self.stage2_encoder = SA(args_s2, input_dim=768)

        # decoders（复用现有 decoder 逻辑）
        decoder_type = str(getattr(args, "decoder_type", "mlp")).lower()
        if decoder_type in ["mlp", "ffn"]:
            self.stage1_decoder = MLPDecoder(args_s1)
            self.stage2_decoder = MLPDecoder(args_s2)
        elif decoder_type in ["transformer", "tf", "transformer_decoder"]:
            self.stage1_decoder = TransformerDecoder(args_s1)
            self.stage2_decoder = TransformerDecoder(args_s2)
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

        # pos_dec：与 DINOSAURpp 一致，用于 sbd_slots
        # 说明：这里每个阶段独立一份 pos_dec（更干净，也避免维度不一致）
        self.pos_dec_s1 = nn.Parameter(torch.Tensor(1, self.token_num, self.slot_dim))
        self.pos_dec_s2 = nn.Parameter(torch.Tensor(1, self.token_num, self.slot_dim))
        init.normal_(self.pos_dec_s1, mean=0.0, std=0.02)
        init.normal_(self.pos_dec_s2, mean=0.0, std=0.02)

        # 背景 slot 的“先验”：通过 stage1 的 ISA 初始化 S_s（背景大、前景小）
        if self.ISA:
            bg_init_scale = float(getattr(args, "two_stage_bg_init_scale", 2.0))
            fg_init_scale = float(getattr(args, "two_stage_fg_init_scale", 0.3))
            bg_init_pos = float(getattr(args, "two_stage_bg_init_pos", 0.0))
            fg_init_pos = float(getattr(args, "two_stage_fg_init_pos", 0.0))
            with torch.no_grad():
                # S_s: (1, 2, 1, 3)
                self.stage1_encoder.S_s.zero_()
                self.stage1_encoder.S_s[:, 0, :, :] = bg_init_scale
                self.stage1_encoder.S_s[:, 1, :, :] = fg_init_scale
                # S_p: (1, 2, 1, 3)
                self.stage1_encoder.S_p.zero_()
                self.stage1_encoder.S_p[:, 0, :, :] = bg_init_pos
                self.stage1_encoder.S_p[:, 1, :, :] = fg_init_pos

        # final linear（与原实现一致）
        self.final_layer_s1 = nn.Linear(self.slot_dim, self.slot_dim)
        self.final_layer_s2 = nn.Linear(self.slot_dim, self.slot_dim)

        # 数值稳定
        self.eps = 1e-8

    def _sbd_slots(self, slots: torch.Tensor, pos_dec: torch.Tensor) -> torch.Tensor:
        # slots: (B, S, D_slot) -> (B, S, token, D_slot) with learned pos_dec
        B, S, D_slot = slots.shape
        x = slots.view(-1, 1, D_slot).tile(1, self.token_num, 1)  # (B*S, token, D_slot)
        pos_embed = pos_dec.expand(x.shape)
        x = x + pos_embed
        return x.view(B, S, self.token_num, D_slot)

    @staticmethod
    def _split_slot_maps(slot_maps: torch.Tensor):
        # slot_maps: (B, S, N, 768+1)
        channels, mask_logits = torch.split(slot_maps, [768, 1], dim=-1)
        return channels, mask_logits.squeeze(-1)  # (B,S,N,768), (B,S,N)

    def _softmax_masks(self, mask_logits: torch.Tensor) -> torch.Tensor:
        # mask_logits: (B,S,N) -> masks: (B,S,N)
        return mask_logits.softmax(dim=1)

    def forward(self, features: torch.Tensor, point_coords: Optional[torch.Tensor] = None):
        B, N, _ = features.shape
        assert N == self.token_num, f"token_num 不一致: N={N}, token_num={self.token_num}"

        if self.ISA:
            assert point_coords is not None, "ISA 两阶段模式需要提供 point_coords"

            # ===== Stage 1: bg/fg =====
            slots_s1, attn_s1 = self.stage1_encoder(features, point_coords)  # (B,2,D), (B,2,N)
            abs_grid = point_coords.unsqueeze(1).expand(B, 2, N, 3)
            rel_grid_s1 = self.stage1_encoder.get_rel_grid(attn_s1, abs_grid)  # (B,2,N,D)
            slot_maps_s1 = self._sbd_slots(slots_s1, self.pos_dec_s1) + rel_grid_s1  # (B,2,N,D)
            slot_maps_s1 = self.stage1_decoder(slot_maps_s1.reshape(B * 2, N, -1)).reshape(B, 2, N, -1)

            ch_s1, logit_s1 = self._split_slot_maps(slot_maps_s1)  # ch:(B,2,N,768), logit:(B,2,N)
            masks_s1 = self._softmax_masks(logit_s1)               # (B,2,N)
            bg_prior = masks_s1[:, 0, :]                           # (B,N)
            fg_prior = masks_s1[:, 1, :]                           # (B,N)

            # ===== Stage 2: foreground decomposition =====
            # 关键：token_weights=fg_prior，使得背景 token 不参与 slot 更新（updates/S_p/S_s 全部只由前景 token 贡献）
            slots_s2, attn_s2 = self.stage2_encoder(features, point_coords, token_weights=fg_prior)
            abs_grid2 = point_coords.unsqueeze(1).expand(B, self.slot_num - 1, N, 3)
            rel_grid_s2 = self.stage2_encoder.get_rel_grid(attn_s2, abs_grid2)  # (B,S-1,N,D)
            slot_maps_s2 = self._sbd_slots(slots_s2, self.pos_dec_s2) + rel_grid_s2
            slot_maps_s2 = self.stage2_decoder(slot_maps_s2.reshape(B * (self.slot_num - 1), N, -1)).reshape(
                B, self.slot_num - 1, N, -1
            )

            ch_s2, logit_s2 = self._split_slot_maps(slot_maps_s2)  # (B,S-1,N,768), (B,S-1,N)
            masks_s2_local = self._softmax_masks(logit_s2)          # (B,S-1,N) 仅在前景 slot 内归一化

            # 将 stage2 masks 限制在前景区域（背景 token 的前景概率为 0）
            masks_fg = masks_s2_local * fg_prior.unsqueeze(1)       # (B,S-1,N)
            masks_bg = bg_prior.unsqueeze(1)                        # (B,1,N)

            # 最终 masks：拼接 bg + fg，然后在 slot 维度重新归一化（保证每个 token 的总和为 1）
            masks_final = torch.cat([masks_bg, masks_fg], dim=1)     # (B,S,N)
            masks_final = masks_final / (masks_final.sum(dim=1, keepdim=True) + self.eps)

            # 最终 channels：背景使用 stage1 的 bg channel；前景使用 stage2 channels
            ch_bg = ch_s1[:, 0:1, :, :]                             # (B,1,N,768)
            ch_final = torch.cat([ch_bg, ch_s2], dim=1)             # (B,S,N,768)

            reconstruction = torch.sum(ch_final * masks_final.unsqueeze(-1), dim=1)  # (B,N,768)

            # slots 输出：bg slot 来自 stage1 的 slot0；fg slots 来自 stage2
            slots_bg = self.final_layer_s1(slots_s1[:, 0:1, :])      # (B,1,D)
            slots_fg = self.final_layer_s2(slots_s2)                 # (B,S-1,D)
            slots_final = torch.cat([slots_bg, slots_fg], dim=1)     # (B,S,D)

            # ===== 可视化缓存（训练/推理不依赖这些字段）=====
            # 让训练脚本在可视化时拿到 stage1/stage2 的分配结果
            self._vis_stage1_masks = masks_s1.detach()        # (B,2,N)
            self._vis_stage2_masks = masks_s2_local.detach()  # (B,S-1,N)
            self._vis_bg_prior = bg_prior.detach()            # (B,N)
            self._vis_fg_prior = fg_prior.detach()            # (B,N)

            return reconstruction, slots_final, masks_final

        # ===== SA（非 ISA）分支：不使用 point_coords =====
        slots_s1 = self.stage1_encoder(features)                     # (B,2,D)
        slot_maps_s1 = self._sbd_slots(slots_s1, self.pos_dec_s1)     # (B,2,N,D)
        slot_maps_s1 = self.stage1_decoder(slot_maps_s1.reshape(B * 2, N, -1)).reshape(B, 2, N, -1)
        ch_s1, logit_s1 = self._split_slot_maps(slot_maps_s1)
        masks_s1 = self._softmax_masks(logit_s1)
        bg_prior = masks_s1[:, 0, :]
        fg_prior = masks_s1[:, 1, :]

        slots_s2 = self.stage2_encoder(features, token_weights=fg_prior)  # (B,S-1,D)
        slot_maps_s2 = self._sbd_slots(slots_s2, self.pos_dec_s2)
        slot_maps_s2 = self.stage2_decoder(slot_maps_s2.reshape(B * (self.slot_num - 1), N, -1)).reshape(
            B, self.slot_num - 1, N, -1
        )
        ch_s2, logit_s2 = self._split_slot_maps(slot_maps_s2)
        masks_s2_local = self._softmax_masks(logit_s2)

        masks_fg = masks_s2_local * fg_prior.unsqueeze(1)
        masks_bg = bg_prior.unsqueeze(1)
        masks_final = torch.cat([masks_bg, masks_fg], dim=1)
        masks_final = masks_final / (masks_final.sum(dim=1, keepdim=True) + self.eps)

        ch_bg = ch_s1[:, 0:1, :, :]
        ch_final = torch.cat([ch_bg, ch_s2], dim=1)
        reconstruction = torch.sum(ch_final * masks_final.unsqueeze(-1), dim=1)

        slots_bg = self.final_layer_s1(slots_s1[:, 0:1, :])
        slots_fg = self.final_layer_s2(slots_s2)
        slots_final = torch.cat([slots_bg, slots_fg], dim=1)

        # 可视化缓存
        self._vis_stage1_masks = masks_s1.detach()
        self._vis_stage2_masks = masks_s2_local.detach()
        self._vis_bg_prior = bg_prior.detach()
        self._vis_fg_prior = fg_prior.detach()
        return reconstruction, slots_final, masks_final


    
