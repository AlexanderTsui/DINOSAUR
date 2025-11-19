import warnings

import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as F
from torch.nn import init
import random
import timm

from sklearn.cluster import AgglomerativeClustering


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



class Decoder(nn.Module):
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

        # 3D坐标编码网络（从2D改为3D）
        self.h = nn.Linear(3, self.slot_dim)  # 用于get_rel_grid
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

        init.normal_(self.S_s, mean=0., std=.02)
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

        # 3D相对坐标编码网络（从2D改为3D）
        self.g = nn.Linear(3, self.slot_dim)  # 用于forward中的相对坐标
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

        # 归一化得到相对坐标（3D）
        rel_grid = (abs_grid - S_p) / (S_s * self.sigma + 1e-8)                 # (B, S, N, 3)
        rel_grid = self.h(rel_grid)                                             # (B, S, N, D_slot) - 通过MLP编码

        return rel_grid


    def forward(self, inputs, point_coords):
        # :arg inputs:              (B, N, D) - 点云特征
        # :arg point_coords:        (B, N, 3) - 点云的3D坐标（已归一化到[-1, 1]）
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
        inputs = self.initial_mlp(inputs).unsqueeze(dim=1)          # (B, 1, N, D_slot)
        inputs = inputs.expand(B, S, N, D_slot)                     # (B, S, N, D_slot)

        # 构建3D abs_grid：从输入的点云坐标
        abs_grid = point_coords.unsqueeze(1)                        # (B, 1, N, 3)
        abs_grid = abs_grid.expand(B, S, N, 3)                      # (B, S, N, 3)

        assert torch.sum(torch.isnan(abs_grid)) == 0, "abs_grid包含NaN"

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
            rel_grid = (abs_grid - S_p) / (S_s * self.sigma + 1e-8)  # (B, S, N, 3) - 3D相对坐标
            k = self.f(self.K(inputs) + self.g(rel_grid))            # (B, S, N, D_slot)
            v = self.f(self.V(inputs) + self.g(rel_grid))            # (B, S, N, D_slot)

            # === Calculate attention ===
            q = self.Q(slots).unsqueeze(dim=-1)                      # (B, S, D_slot, 1)

            dots = torch.einsum('bsdi,bsjd->bsj', q, k)              # (B, S, D_slot, 1) x (B, S, N, D_slot) -> (B, S, N)
            dots *=  self.scale                                      # (B, S, N)
            attn = dots.softmax(dim=1) + epsilon                     # (B, S, N) - softmax over slots

            # === Weighted mean ===
            attn = attn / attn.sum(dim=-1, keepdim=True)             # (B, S, N) - 归一化
            attn = attn.unsqueeze(dim=2)                             # (B, S, 1, N)
            updates = torch.einsum('bsjd,bsij->bsd', v, attn)        # (B, S, N, D_slot) x (B, S, 1, N) -> (B, S, D_slot)

            # === Update S_p and S_s (3D) ===
            S_p = torch.einsum('bsjd,bsij->bsd', abs_grid, attn)     # (B, S, N, 3) x (B, S, 1, N) -> (B, S, 3)
            S_p = S_p.unsqueeze(dim=2)                               # (B, S, 1, 3)

            values_ss = torch.pow(abs_grid - S_p, 2)                 # (B, S, N, 3)
            S_s = torch.einsum('bsjd,bsij->bsd', values_ss, attn)    # (B, S, N, 3) x (B, S, 1, N) -> (B, S, 3)
            S_s = torch.sqrt(S_s + 1e-8)                             # (B, S, 3) - 添加epsilon避免sqrt(0)
            S_s = S_s.unsqueeze(dim=2)                               # (B, S, 1, 3)

            # === Update slots (与坐标维度无关，保持不变) ===
            if t != self.iters:
                slots = self.gru(
                    updates.reshape(-1, self.slot_dim),
                    slots_prev.reshape(-1, self.slot_dim))

                slots = slots.reshape(B, -1, self.slot_dim)
                slots = self.mlp(slots)

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

    def forward(self, inputs):
        # :arg inputs:              (B, token, D)
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

        self.slot_decoder = Decoder(args)

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


    
