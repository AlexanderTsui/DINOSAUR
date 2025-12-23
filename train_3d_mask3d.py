"""
DINOSAUR 3D训练脚本 (Encoder 特征输入)

训练流程:
1. S3DIS数据加载 → 重采样到80000点
2. Mask3D提取点级特征f_point (冻结)
3. FPS采样固定数量超点
4. 特征投影 96→768维
5. DINOSAUR ISA处理
6. 4项损失计算
7. TensorBoard监控 + HTML可视化
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from scipy.spatial import cKDTree

# TensorBoard导入（可选）
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    print("警告: TensorBoard未安装，将跳过TensorBoard日志")
    HAS_TENSORBOARD = False
    SummaryWriter = None

# Plotly（生成交互式HTML）
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly.colors import qualitative as plotly_qualitative
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ==================== 路径设置 ====================
# current_dir: DINOSAUR 目录（包含 data/, models/, utils/ 子目录）
# 使用绝对导入避免与 src/models 冲突
current_dir = os.path.dirname(os.path.abspath(__file__))

# 使用 importlib 进行绝对路径导入，避免 sys.path 污染导致的模块冲突
import importlib.util

def _import_module_from_path(module_name, file_path):
    """从指定路径导入模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# 导入 DINOSAUR/data 下的数据集模块
_s3dis_dataset = _import_module_from_path(
    'dinosaur_s3dis_dataset',
    os.path.join(current_dir, 'data', 's3dis_dataset_mask3d.py')
)
S3DISMask3DDataset = _s3dis_dataset.S3DISMask3DDataset
collate_fn_mask3d = _s3dis_dataset.collate_fn_mask3d

_scannet_dataset = _import_module_from_path(
    'dinosaur_scannet_dataset',
    os.path.join(current_dir, 'data', 'scannet_dataset.py')
)
ScanNetDataset = _scannet_dataset.ScanNetDataset
collate_fn_scannet = _scannet_dataset.collate_fn_scannet

# 导入 DINOSAUR/models 下的模块
_mask3d_wrapper = _import_module_from_path(
    'dinosaur_mask3d_wrapper',
    os.path.join(current_dir, 'models', 'mask3d_wrapper.py')
)
create_mask3d_dinosaur_model = _mask3d_wrapper.create_mask3d_dinosaur_model
create_logosp_dinosaur_model = _mask3d_wrapper.create_logosp_dinosaur_model
create_concerto_dinosaur_model = _mask3d_wrapper.create_concerto_dinosaur_model

_losses = _import_module_from_path(
    'dinosaur_losses',
    os.path.join(current_dir, 'models', 'losses.py')
)
DINOSAURLoss = _losses.DINOSAURLoss

# 导入可视化工具
_visualizer = _import_module_from_path(
    'dinosaur_visualizer',
    os.path.join(current_dir, 'utils', 'visualizer.py')
)
visualize_slot_assignment = _visualizer.visualize_slot_assignment
visualize_reconstruction_error = _visualizer.visualize_reconstruction_error
visualize_slot_statistics = _visualizer.visualize_slot_statistics


def _rgb_to_plotly_colors(rgb_array: np.ndarray):
    """将归一化RGB转换为Plotly支持的'rgb(r,g,b)'字符串列表。"""
    if rgb_array.size == 0:
        return []
    
    rgb = np.asarray(rgb_array, dtype=np.float32)
    if rgb.min() < 0:
        rgb = (rgb + 1.0) / 2.0  # [-1,1] -> [0,1]
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb_int = (rgb * 255).astype(np.uint8)
    return [f'rgb({r},{g},{b})' for r, g, b in rgb_int]


def save_slot_assignment_html(original_xyz, original_rgb, slot_xyz, slot_assignments,
                              num_slots, output_path, point_size=2.0, slot_size=4.0):
    """生成交互式3D HTML，左侧原始点云，右侧slot绑定结果。"""
    import signal
    import sys
    
    if not HAS_PLOTLY:
        print("[可视化] 未检测到Plotly，跳过HTML生成")
        return
    
    if original_xyz.size == 0 or slot_xyz.size == 0:
        print("[可视化] 数据为空，跳过HTML生成")
        return
    
    print(f"[可视化] 开始生成HTML: 原始点数={len(original_xyz)}, 超点数={len(slot_xyz)}")
    sys.stdout.flush()
    
    # 大幅减少点数以避免卡死（临时诊断）
    max_vis_points = 10000  # 原来是50000，减少到10000
    if len(original_xyz) > max_vis_points:
        print(f"[可视化] 下采样: {len(original_xyz)} -> {max_vis_points}")
        sys.stdout.flush()
        idx = np.random.choice(len(original_xyz), max_vis_points, replace=False)
        original_xyz = original_xyz[idx]
        original_rgb = original_rgb[idx]
    
    print("[可视化] 生成颜色...")
    sys.stdout.flush()
    
    # 原始点云颜色
    orig_colors = _rgb_to_plotly_colors(original_rgb)
    
    # Slot颜色（重复使用调色板）
    color_palette = getattr(plotly_qualitative, 'Dark24', plotly_qualitative.Plotly)
    repeats = (num_slots + len(color_palette) - 1) // len(color_palette)
    expanded_palette = (color_palette * repeats)[:num_slots]
    slot_assignments = np.asarray(slot_assignments, dtype=np.int64)
    slot_point_colors = [expanded_palette[int(idx)] for idx in slot_assignments]
    
    print("[可视化] 创建Plotly图表...")
    sys.stdout.flush()
    
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('Original Point Cloud', 'Slot Assignments (Superpoints)')
    )
    
    print("[可视化] 添加原始点云trace...")
    sys.stdout.flush()
    
    fig.add_trace(
        go.Scatter3d(
            x=original_xyz[:, 0],
            y=original_xyz[:, 1],
            z=original_xyz[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=orig_colors,
                opacity=0.85
            ),
            name='Input'
        ),
        row=1,
        col=1
    )
    
    print("[可视化] 添加Slot点云trace...")
    sys.stdout.flush()
    
    fig.add_trace(
        go.Scatter3d(
            x=slot_xyz[:, 0],
            y=slot_xyz[:, 1],
            z=slot_xyz[:, 2],
            mode='markers',
            marker=dict(
                size=slot_size,
                color=slot_point_colors,
                opacity=0.9
            ),
            name='Slots'
        ),
        row=1,
        col=2
    )
    
    scene_layout = dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
        aspectmode='data'
    )
    fig.update_layout(
        height=600,
        width=1200,
        template='plotly_white',
        showlegend=False,
    )
    fig.update_scenes(scene_layout, row=1, col=1)
    fig.update_scenes(scene_layout, row=1, col=2)
    
    print(f"[可视化] 写入HTML文件: {output_path}")
    sys.stdout.flush()
    
    # 使用超时机制防止卡死
    def timeout_handler(signum, frame):
        raise TimeoutError("HTML写入超时!")
    
    try:
        # 设置60秒超时（仅Linux有效）
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)
        
        fig.write_html(output_path)
        
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # 取消超时
            
        print(f"[可视化] HTML已保存: {output_path}")
    except TimeoutError:
        print("[可视化] HTML写入超时，跳过!")
    except Exception as e:
        print(f"[可视化] HTML写入失败: {e}")
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
    
    sys.stdout.flush()


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_distributed():
    """设置分布式训练"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        
        return rank, world_size, local_rank
    else:
        # 单机单卡
        return 0, 1, 0


def _ddp_spawn_worker(local_rank: int, world_size: int, args, config):
    """
    当用户未用 torchrun 启动时，脚本内部自动多进程启动 DDP。
    - 依赖 CUDA_VISIBLE_DEVICES 已在主进程设置好（可用 config.gpu_ids）
    - 单机单节点：用 env:// 初始化，自动补齐 MASTER_ADDR/PORT
    """
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    # 允许在 config.distributed.master_port 指定端口
    dist_cfg = (config.get('distributed', {}) or {})
    os.environ.setdefault('MASTER_PORT', str(dist_cfg.get('master_port', 29500)))

    os.environ['RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)

    _main_impl(args, config)


def _main_impl(args, config):
    """真正的训练入口（单进程或 DDP worker 进程都会走这里）。"""
    if args.test_run:
        print("\n⚠️  测试运行模式：仅训练2个epoch")
        config['train']['epochs'] = 2
        config['validation']['val_interval'] = 1
        config['checkpoint']['save_interval'] = 1

    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # 分布式设置
    rank, world_size, local_rank = setup_distributed()

    if config['train'].get('detect_anomaly', False):
        torch.autograd.set_detect_anomaly(True)
        if rank == 0:
            print("[调试] 已启用Autograd异常检测")

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"DINOSAUR 3D训练 (Mask3D特征)")
        print(f"{'='*60}")
        print(f"配置: {args.config}")
        print(f"设备数: {world_size}")
        print(f"Slot数量: {config['model']['num_slots']}")
        print(f"超点数量: {config['model']['num_superpoints']}")

    # 创建输出目录
    output_dir = config['checkpoint']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)

    # TensorBoard
    writer = None
    if rank == 0 and HAS_TENSORBOARD:
        log_dir = os.path.join(output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard日志: {log_dir}")

    # 创建数据集（支持 S3DIS / ScanNet）
    dataset_name = config['data'].get('dataset', 's3dis').lower()
    train_scope = str(config['data'].get('train_scope', 'train')).lower()
    if dataset_name == 's3dis':
        train_dataset = S3DISMask3DDataset(
            root_dir=config['data']['s3dis_root'],
            areas=config['data']['train_areas'],
            max_points=config['data']['max_points'],
            augment=True,
            aug_config=config['augmentation']
        )

        val_dataset = S3DISMask3DDataset(
            root_dir=config['data']['s3dis_root'],
            areas=config['data']['val_areas'],
            max_points=config['data']['max_points'],
            augment=False
        )
        collate_fn = collate_fn_mask3d
    elif dataset_name == 'scannet':
        # ScanNet: 强制使用官方 train/val 划分；自监督可选 train / val / train+val
        allowed_scopes = {'train', 'val', 'trainval', 'all'}
        if train_scope not in allowed_scopes:
            raise ValueError(f"不支持的train_scope: {train_scope}，可选: {allowed_scopes}")

        def build_scannet(split, for_training):
            return ScanNetDataset(
                root_dir=config['data']['scannet_root'],
                split=split,  # 官方划分
                max_points=config['data']['max_points'],
                augment=for_training,
                aug_config=config['augmentation']
            )

        if train_scope == 'train':
            train_dataset = build_scannet('train', True)
            train_split_desc = '官方 train'
        elif train_scope == 'val':
            train_dataset = build_scannet('val', True)
            train_split_desc = '官方 val (自监督)'
        else:  # trainval / all
            train_dataset = ConcatDataset([
                build_scannet('train', True),
                build_scannet('val', True),
            ])
            train_split_desc = '官方 train+val (自监督)'

        # 验证集始终使用官方 val，保持评估一致
        val_dataset = build_scannet('val', False)
        collate_fn = collate_fn_scannet
    else:
        raise ValueError(f"未知数据集: {dataset_name}")

    # 创建DataLoader（DDP: DistributedSampler）
    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False
        )
    train_loader_kwargs = dict(
        batch_size=config['train']['batch_size_per_gpu'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config['train']['num_workers'],
        pin_memory=config['train']['pin_memory'],
        collate_fn=collate_fn
    )
    if config['train']['num_workers'] > 0:
        train_loader_kwargs['persistent_workers'] = config['train'].get('persistent_workers', False)
        train_loader_kwargs['prefetch_factor'] = config['train'].get('prefetch_factor', 2)
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    # 准备可视化样本
    num_vis = config.get('visualization', {}).get('num_vis_samples', 3)
    vis_samples = [val_dataset[i] for i in range(min(num_vis, len(val_dataset)))]

    if rank == 0:
        print(f"\n数据集:")
        print(f"  训练: {len(train_dataset)} 个房间")
        print(f"  验证: {len(val_dataset)} 个房间")
        if dataset_name == 'scannet':
            print(f"  训练划分: {train_split_desc}（严格使用官方train/val列表）")

    # 创建模型（支持 Mask3D / LogoSP / Concerto）
    backbone = config['model'].get('backbone', 'mask3d').lower()
    if backbone == 'mask3d':
        model = create_mask3d_dinosaur_model(config).cuda()
    elif backbone == 'logosp':
        model = create_logosp_dinosaur_model(config).cuda()
    elif backbone == 'concerto':
        model = create_concerto_dinosaur_model(config).cuda()
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=config['distributed']['find_unused_parameters']
        )

    # 损失函数
    criterion = DINOSAURLoss(config['loss'])

    # 检测是否使用Two-Stage DINOSAUR（用于对比损失）
    target_model = model.module if world_size > 1 else model
    use_two_stage = hasattr(target_model, 'dinosaur') and target_model.dinosaur.__class__.__name__ == 'TwoStageDINOSAURpp'
    if rank == 0:
        print(f"  Two-Stage模式: {'是' if use_two_stage else '否'}")
        if use_two_stage:
            print(f"  → 对比损失将包含前景-背景对比项")

    # 优化器
    optimizer = torch.optim.AdamW(
        model.get_trainable_params() if world_size == 1 else model.module.get_trainable_params(),
        lr=config['train']['optimizer']['lr'],
        weight_decay=config['train']['optimizer']['weight_decay'],
        betas=tuple(config['train']['optimizer']['betas'])
    )

    # 学习率调度器 (PolyLR)
    total_iters = len(train_loader) * config['train']['epochs']
    warmup_iters = len(train_loader) * config['train']['warmup_epochs']

    def lr_lambda(current_iter):
        # 重要：恢复训练/更换 world_size 后，scheduler.last_epoch 可能 > total_iters，
        # 若不 clamp，(1-progress) 会变成负数，负数的非整数次幂会产生复数学习率 -> TensorBoard 报错。
        if warmup_iters <= 0:
            warmup_scale = 1.0
        else:
            warmup_scale = float(current_iter) / float(warmup_iters)
            warmup_scale = 0.0 if warmup_scale < 0.0 else (1.0 if warmup_scale > 1.0 else warmup_scale)

        if current_iter < warmup_iters:
            return warmup_scale

        denom = float(max(total_iters - warmup_iters, 1))
        progress = float(current_iter - warmup_iters) / denom
        progress = 0.0 if progress < 0.0 else (1.0 if progress > 1.0 else progress)

        base = 1.0 - progress
        if base < 0.0:
            base = 0.0
        power = float(config['train']['scheduler']['power'])
        return base ** power

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 混合精度
    scaler = GradScaler(enabled=config['train']['use_amp'])

    # 可选恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    resume_path = args.resume
    if not resume_path:
        resume_path = (config.get('train', {}) or {}).get('resume_from', '') or ''
    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cuda')
        target_model = model.module if world_size > 1 else model
        target_model.projector.load_state_dict(checkpoint['model_state_dict']['projector'])
        target_model.dinosaur.load_state_dict(checkpoint['model_state_dict']['dinosaur'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_val_loss = checkpoint.get('val_loss', best_val_loss)
        start_epoch = checkpoint.get('epoch', -1) + 1
        if rank == 0:
            print(f"[恢复] 从epoch {checkpoint.get('epoch', '未知')} 继续训练，best_val_loss={best_val_loss:.6f}")
            print(f"[恢复] checkpoint: {resume_path}")

    # ==================== 推理导出（不训练） ====================
    infer_cfg = config.get('inference', {})
    if infer_cfg.get('export_ply', False):
        ckpt_path = infer_cfg.get('checkpoint_path', '') or os.path.join(output_dir, 'best_model.pth')
        sample_index = int(infer_cfg.get('sample_index', 0))
        out_ply = infer_cfg.get('output_ply', os.path.join(output_dir, f'slot_assignment_{sample_index}.ply'))

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"未找到推理权重: {ckpt_path}")

        print(f"\n[Infer] 仅推理导出模式")
        print(f"[Infer] checkpoint: {ckpt_path}")
        print(f"[Infer] sample_index: {sample_index}")
        print(f"[Infer] output_ply: {out_ply}")

        ckpt = torch.load(ckpt_path, map_location='cuda')
        target_model = model.module if world_size > 1 else model
        target_model.projector.load_state_dict(ckpt['model_state_dict']['projector'])
        target_model.dinosaur.load_state_dict(ckpt['model_state_dict']['dinosaur'])
        target_model.eval()

        if sample_index < 0 or sample_index >= len(val_dataset):
            raise IndexError(f"sample_index 越界: {sample_index}, val_dataset len={len(val_dataset)}")
        sample = val_dataset[sample_index]
        export_slot_assignment_ply(target_model, sample, config, out_ply)
        return

    if rank == 0:
        print(f"\n模型创建完成")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"可训练参数: {trainable_params:,}")

    # 训练循环
    start_time = time.time()

    train_cfg = config.get('train', {}) or {}
    # 预训练（feat_rec only）两阶段开关：
    # - enable_pretrain_featrec_only: False -> 完全关闭该模式（即使 pretrain_featrec_only_epochs>0）
    # - pretrain_featrec_only_epochs<=0 -> 同样关闭
    pretrain_enabled = bool(train_cfg.get('enable_pretrain_featrec_only', True))
    pretrain_epochs_cfg = int(train_cfg.get('pretrain_featrec_only_epochs', 0) or 0)
    pretrain_epochs = pretrain_epochs_cfg if pretrain_enabled else 0
    stop_after_pretrain = bool(train_cfg.get('stop_after_pretrain', False)) if pretrain_enabled else False
    last_val_loss = float('nan')
    early_stop = False

    for epoch in range(start_epoch, config['train']['epochs']):
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{config['train']['epochs']}")
            print(f"{'='*60}")

        # 训练
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            epoch, config, writer, rank
        )

        if rank == 0:
            print(f"\n训练损失: {train_losses['total']:.6f}")
            if writer is not None:
                writer.add_scalar('Epoch/train_loss', train_losses['total'], epoch)

        # 验证
        if (epoch + 1) % config['validation']['val_interval'] == 0:
            val_loss = validate(
                model, val_loader, criterion, epoch, config, writer, rank
            )
            last_val_loss = float(val_loss)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            if rank == 0:
                save_checkpoint(
                    epoch, model.module if world_size > 1 else model,
                    optimizer, scheduler, val_loss, config, output_dir, is_best
                )

        # === 预训练阶段结束：强制保存基线并退出 ===
        if pretrain_epochs > 0 and stop_after_pretrain and (epoch + 1) >= pretrain_epochs:
            if rank == 0:
                if np.isnan(last_val_loss):
                    try:
                        last_val_loss = float(validate(model, val_loader, criterion, epoch, config, writer, rank))
                    except Exception:
                        last_val_loss = float('nan')

                base_path = os.path.join(output_dir, f'pretrain_featrec_only_epoch_{epoch+1:03d}.pth')
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': {
                        'projector': (model.module if world_size > 1 else model).projector.state_dict(),
                        'dinosaur': (model.module if world_size > 1 else model).dinosaur.state_dict()
                    },
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': last_val_loss,
                    'config': config
                }
                torch.save(ckpt, base_path)
                print(f"\n[Pretrain] 已到达预训练轮数 {pretrain_epochs}，基线权重已保存: {base_path}")
                print("[Pretrain] 根据该 checkpoint 调整 config 后可继续训练（设置 train.resume_from 指向该文件）")
            early_stop = True
            break

        # 可视化
        if rank == 0 and (epoch + 1) % config['checkpoint']['save_interval'] == 0:
            visualize_samples(
                model.module if world_size > 1 else model,
                vis_samples, epoch, config,
                os.path.join(output_dir, 'visualizations')
            )

    # 训练完成（或预训练提前退出）
    total_time = time.time() - start_time
    if rank == 0:
        print(f"\n{'='*60}")
        if early_stop:
            print(f"训练提前结束（预训练阶段完成）!")
        else:
            print(f"训练完成!")
        print(f"总时间: {total_time/3600:.2f} 小时")
        print(f"最佳验证损失: {best_val_loss:.6f}")
        print(f"{'='*60}\n")
        if writer is not None:
            writer.close()

    if world_size > 1:
        dist.destroy_process_group()


def _apply_loss_warmup(criterion, config, epoch: int):
    """
    按 epoch 对各项 loss 权重做 warmup（线性 ramp），让模型先学会重建再逐步细化。

    约定（来自 config['loss']）：
      loss:
        weights: {feat_rec, compact, entropy, min_usage, smooth, cons}
        warmup:
          enabled: True
          log_weights: True
          items:
            compact:
              enabled: True
              start_epoch: 10
              warmup_epochs: 20
              start_weight: 0.0
              end_weight: 0.5   # 可选，不写则用 loss.weights.compact
    """
    loss_cfg = config.get('loss', {}) or {}
    target_weights = (loss_cfg.get('weights', {}) or {}).copy()

    warm = (loss_cfg.get('warmup', {}) or {})
    if not bool(warm.get('enabled', False)):
        # 直接使用目标权重
        w_now = {k: float(v) for k, v in target_weights.items()}
    else:
        items = warm.get('items', {}) or {}

        def _lerp(a: float, b: float, t: float) -> float:
            return a + (b - a) * t

        w_now = {}
        for k, w_tgt in target_weights.items():
            cfg_k = items.get(k, {}) or {}
            if cfg_k.get('enabled', True) is False:
                w_now[k] = 0.0
                continue

            start_epoch = int(cfg_k.get('start_epoch', 0))
            warmup_epochs = int(cfg_k.get('warmup_epochs', 0))
            start_weight = float(cfg_k.get('start_weight', 0.0))
            end_weight = float(cfg_k.get('end_weight', w_tgt))

            if epoch < start_epoch:
                w_now[k] = start_weight
            else:
                if warmup_epochs <= 0:
                    w_now[k] = end_weight
                else:
                    t = (epoch - start_epoch) / float(warmup_epochs)
                    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                    w_now[k] = _lerp(start_weight, end_weight, t)

    # 写回到 criterion（避免改 loss.py，且对反传最直接）
    # DINOSAURLoss 内部字段名：w_feat_rec/w_compact/w_entropy/w_min_usage/w_smooth/w_cons/w_diversity
    setattr(criterion, 'w_feat_rec', float(w_now.get('feat_rec', getattr(criterion, 'w_feat_rec', 1.0))))
    setattr(criterion, 'w_compact', float(w_now.get('compact', getattr(criterion, 'w_compact', 0.0))))
    setattr(criterion, 'w_entropy', float(w_now.get('entropy', getattr(criterion, 'w_entropy', 0.0))))
    setattr(criterion, 'w_min_usage', float(w_now.get('min_usage', getattr(criterion, 'w_min_usage', 0.0))))
    setattr(criterion, 'w_smooth', float(w_now.get('smooth', getattr(criterion, 'w_smooth', 0.0))))
    setattr(criterion, 'w_cons', float(w_now.get('cons', getattr(criterion, 'w_cons', 0.0))))
    setattr(criterion, 'w_diversity', float(w_now.get('diversity', getattr(criterion, 'w_diversity', 0.0))))

    return w_now


def _maybe_force_featrec_only(criterion, config, epoch: int, rank: int, writer):
    """
    前若干个 epoch 仅开启重建损失（feat_rec），用于稳定预训练/对比实验。
    由 config['train']['pretrain_featrec_only_epochs'] 控制：
      - <=0: 关闭
      - >0 且 epoch < pretrain_epochs: 强制 w_compact/w_entropy/w_min_usage/w_smooth/w_cons = 0
    """
    train_cfg = config.get('train', {}) or {}
    if bool(train_cfg.get('enable_pretrain_featrec_only', True)) is False:
        return False
    pretrain_epochs = int(train_cfg.get('pretrain_featrec_only_epochs', 0) or 0)
    if pretrain_epochs <= 0 or epoch >= pretrain_epochs:
        return False

    # 保留当前 feat_rec 权重（可能来自 warmup），其它全部置零
    setattr(criterion, 'w_compact', 0.0)
    setattr(criterion, 'w_entropy', 0.0)
    setattr(criterion, 'w_min_usage', 0.0)
    setattr(criterion, 'w_smooth', 0.0)
    setattr(criterion, 'w_cons', 0.0)
    setattr(criterion, 'w_diversity', 0.0)

    if rank == 0:
        print(f"[Pretrain] epoch={epoch} < {pretrain_epochs}: 强制仅使用 feat_rec，其它 loss 权重置 0")
        if writer is not None:
            writer.add_scalar('Pretrain/featrec_only', 1.0, epoch)
    return True


def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, config, writer, rank):
    """训练一个epoch"""
    model.train()

    # 检测是否使用Two-Stage DINOSAUR（用于对比损失）
    target_model = model.module if hasattr(model, 'module') else model
    use_two_stage = hasattr(target_model, 'dinosaur') and target_model.dinosaur.__class__.__name__ == 'TwoStageDINOSAURpp'

    # DDP: 让 DistributedSampler 每个 epoch 产生不同 shuffle
    if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)
    # Mask3D保持eval模式
    if hasattr(model, 'mask3d'):
        model.mask3d.eval()
    elif hasattr(model, 'module') and hasattr(model.module, 'mask3d'):
        model.module.mask3d.eval()
    # Concerto保持eval模式（冻结特征提取器）
    if hasattr(model, 'concerto'):
        model.concerto.eval()
    elif hasattr(model, 'module') and hasattr(model.module, 'concerto'):
        model.module.concerto.eval()
    
    # 预缓存可训练参数
    if hasattr(model, 'get_trainable_params'):
        trainable_params = model.get_trainable_params()
        named_trainable_params = [
            (name, param) for name, param in model.named_parameters()
            if param.requires_grad
        ]
    else:
        trainable_params = model.module.get_trainable_params()
        named_trainable_params = [
            (name, param) for name, param in model.module.named_parameters()
            if param.requires_grad
        ]
    
    total_losses = {
        'total': 0.0,
        'feat_rec': 0.0,
        'compact': 0.0,
        'entropy': 0.0,
        'min_usage': 0.0,
        'smooth': 0.0,
        'cons': 0.0,
        'diversity': 0.0,
        # two-stage: stage1 bg/fg 分离正则（若不是 two-stage 或未启用，会一直为 0）
        'stage1_bgfg_kl': 0.0,
        'stage1_bg_mean': 0.0,
        'stage1_fg_mean': 0.0,
    }

    # loss warmup：每个 epoch 开始时更新权重（数值更稳、避免突然打开导致震荡）
    w_now = _apply_loss_warmup(criterion, config, epoch)
    _maybe_force_featrec_only(criterion, config, epoch, rank, writer)
    if rank == 0:
        warm = (config.get('loss', {}) or {}).get('warmup', {}) or {}
        if bool(warm.get('enabled', False)):
            print(
                f"[LossWarmup] epoch={epoch} "
                f"w_feat_rec={w_now.get('feat_rec', 0.0):.4f}, "
                f"w_compact={w_now.get('compact', 0.0):.4f}, "
                f"w_entropy={w_now.get('entropy', 0.0):.4f}, "
                f"w_min_usage={w_now.get('min_usage', 0.0):.4f}, "
                f"w_smooth={w_now.get('smooth', 0.0):.4f}, "
                f"w_cons={w_now.get('cons', 0.0):.4f}, "
                f"w_diversity={w_now.get('diversity', 0.0):.4f}"
            )
            if writer is not None and bool(warm.get('log_weights', True)):
                writer.add_scalar('LossWeight/feat_rec', float(w_now.get('feat_rec', 0.0)), epoch)
                writer.add_scalar('LossWeight/compact', float(w_now.get('compact', 0.0)), epoch)
                writer.add_scalar('LossWeight/entropy', float(w_now.get('entropy', 0.0)), epoch)
                writer.add_scalar('LossWeight/min_usage', float(w_now.get('min_usage', 0.0)), epoch)
                writer.add_scalar('LossWeight/smooth', float(w_now.get('smooth', 0.0)), epoch)
                writer.add_scalar('LossWeight/cons', float(w_now.get('cons', 0.0)), epoch)
                writer.add_scalar('LossWeight/diversity', float(w_now.get('diversity', 0.0)), epoch)
    
    if rank == 0:
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    else:
        pbar = train_loader
    
    for iter_idx, batch in enumerate(pbar):
        xyz = batch['xyz'].cuda()
        rgb = batch['rgb'].cuda()
        
        optimizer.zero_grad()
        
        # 混合精度前向传播
        with autocast(enabled=config['train']['use_amp']):
            try:
                reconstruction, slots, masks, sp_feats_proj, sampled_coords = model(xyz, rgb)
                
                # 检查NaN和Inf
                has_nan = (
                    torch.isnan(reconstruction).any() or 
                    torch.isnan(slots).any() or 
                    torch.isnan(masks).any() or
                    torch.isnan(sp_feats_proj).any()
                )
                has_inf = (
                    torch.isinf(reconstruction).any() or 
                    torch.isinf(slots).any() or 
                    torch.isinf(masks).any() or
                    torch.isinf(sp_feats_proj).any()
                )
                
                if has_nan or has_inf:
                    print(f"[警告] Epoch {epoch}, Iter {iter_idx}: 检测到NaN/Inf，跳过此batch")
                    optimizer.zero_grad()
                    continue
                
                # 计算损失（smooth 可用 sampled_coords）
                loss, loss_dict = criterion(reconstruction, sp_feats_proj, slots, masks, sampled_coords, use_two_stage=use_two_stage)

                # ==================== two-stage: 额外约束 stage1 不塌缩到单个 slot ====================
                # 现象定位：如果训练只靠 feat_rec（且 compact/entropy/min_usage 等为 0），
                # 两阶段中的 stage1 很容易退化为“一个 slot 吃掉所有 token”，无法产生 fg/bg 分离。
                # 这里加入一个轻量的使用率先验：KL( mean_n p(slot|n) || [p_bg, 1-p_bg] )
                try:
                    dinosaur = getattr(model, 'dinosaur', None)
                    if dinosaur is None and hasattr(model, 'module'):
                        dinosaur = getattr(model.module, 'dinosaur', None)

                    has_stage1 = (
                        dinosaur is not None and
                        hasattr(dinosaur, '_vis_stage1_masks')
                    )
                    if has_stage1:
                        loss_cfg = (config.get('loss', {}) or {})
                        w_stage1 = float((loss_cfg.get('weights', {}) or {}).get('stage1_bgfg_kl', 0.05))
                        if w_stage1 > 0:
                            masks_s1 = dinosaur._vis_stage1_masks  # (B,2,M)
                            if isinstance(masks_s1, torch.Tensor) and masks_s1.dim() == 3 and masks_s1.shape[1] == 2:
                                eps = float((loss_cfg.get('params', {}) or {}).get('eps', 1e-8))
                                # 背景占比目标：优先读 stage1 专用参数；否则用旧的 bg_area_target；否则默认 0.7
                                params = (loss_cfg.get('params', {}) or {})
                                if 'stage1_bg_area_target' in params:
                                    p_bg = float(params.get('stage1_bg_area_target', 0.7))
                                else:
                                    p_bg = float(params.get('bg_area_target', 0.7))
                                p_bg = 0.7 if (not np.isfinite(p_bg)) else p_bg
                                p_bg = 0.05 if p_bg < 0.05 else (0.95 if p_bg > 0.95 else p_bg)
                                prior = torch.tensor([p_bg, 1.0 - p_bg], device=masks_s1.device, dtype=masks_s1.dtype)
                                prior = prior / (prior.sum() + eps)

                                q = masks_s1.mean(dim=2)  # (B,2)
                                q = q / (q.sum(dim=1, keepdim=True) + eps)
                                q = torch.clamp(q, min=eps, max=1.0)
                                # KL(q || prior)
                                kl = (q * (torch.log(q + eps) - torch.log(prior + eps))).sum(dim=1).mean()

                                loss = loss + w_stage1 * kl
                                # 更新 loss_dict（用于日志）
                                loss_dict['stage1_bgfg_kl'] = float(kl.detach().item())
                                loss_dict['stage1_bg_mean'] = float(masks_s1[:, 0, :].mean().detach().item())
                                loss_dict['stage1_fg_mean'] = float(masks_s1[:, 1, :].mean().detach().item())
                                # total 需要反映“加过正则”的最终 loss
                                loss_dict['total'] = float(loss.detach().item())
                except Exception:
                    # 训练不中断：two-stage 诊断项失败则跳过
                    pass
                
                # 检查loss是否异常
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
                    print(f"[警告] Epoch {epoch}, Iter {iter_idx}: Loss异常 ({loss.item()})，跳过此batch")
                    optimizer.zero_grad()
                    continue
                    
            except (AssertionError, RuntimeError) as e:
                print(f"[错误] Epoch {epoch}, Iter {iter_idx}: {e}")
                print("[跳过此batch并继续训练]")
                optimizer.zero_grad()
                continue
        
        # 反向传播
        try:
            scaler.scale(loss).backward()
        except RuntimeError as e:
            print(f"[错误] Epoch {epoch}, Iter {iter_idx}: 反向传播失败: {e}")
            optimizer.zero_grad()
            continue
        
        # Unscale梯度用于裁剪
        scaler.unscale_(optimizer)
        
        # 检查梯度中的NaN/Inf
        has_grad_nan = any(torch.isnan(p.grad).any() if p.grad is not None else False for p in trainable_params)
        has_grad_inf = any(torch.isinf(p.grad).any() if p.grad is not None else False for p in trainable_params)
        
        if has_grad_nan or has_grad_inf:
            print(f"[警告] Epoch {epoch}, Iter {iter_idx}: 梯度包含NaN/Inf，跳过更新")
            
            if rank == 0 and config['train'].get('log_grad_details_on_nan', True):
                max_report = config['train'].get('grad_debug_max_params', 5)
                reported = 0
                for name, param in named_trainable_params:
                    if param.grad is None:
                        continue
                    grad = param.grad
                    grad_nan = torch.isnan(grad).any()
                    grad_inf = torch.isinf(grad).any()
                    if grad_nan or grad_inf:
                        print(f"  - {name}: NaN={grad_nan}, Inf={grad_inf}")
                        reported += 1
                        if reported >= max_report:
                            break
            optimizer.zero_grad()
            # 重置scaler状态，避免下次unscale_报错
            scaler.update()
            continue
        
        # 梯度裁剪
        try:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                config['train']['grad_clip_norm']
            )
        except RuntimeError as e:
            print(f"[错误] Epoch {epoch}, Iter {iter_idx}: 梯度裁剪失败: {e}")
            optimizer.zero_grad()
            continue
        
        # 检查梯度异常
        if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm > 100:
            print(f"[警告] Epoch {epoch}, Iter {iter_idx}: 梯度异常 (norm={grad_norm})，跳过更新")
            optimizer.zero_grad()
            # 重置scaler状态
            scaler.update()
            continue
        
        # 优化器步进
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # 累积损失
        for key in total_losses.keys():
            total_losses[key] += loss_dict.get(key, 0.0)
        
        # 日志记录
        if rank == 0:
            if iter_idx % config['log_interval'] == 0:
                # 兜底：避免异常情况下学习率变成复数导致 TensorBoard 崩溃
                current_lr = optimizer.param_groups[0]['lr']
                if isinstance(current_lr, complex):
                    current_lr = float(current_lr.real)
                else:
                    current_lr = float(current_lr)
                pbar.set_postfix({
                    'loss': f"{loss_dict['total']:.4f}",
                    'feat_rec': f"{loss_dict['feat_rec']:.4f}",
                    'lr': f"{current_lr:.6f}"
                })
                
                # TensorBoard
                if writer is not None:
                    global_step = epoch * len(train_loader) + iter_idx
                    writer.add_scalar('Train/total_loss', loss_dict['total'], global_step)
                    writer.add_scalar('Train/feat_rec', loss_dict['feat_rec'], global_step)
                    writer.add_scalar('Train/compact', loss_dict['compact'], global_step)
                    writer.add_scalar('Train/entropy', loss_dict['entropy'], global_step)
                    writer.add_scalar('Train/min_usage', loss_dict['min_usage'], global_step)
                    writer.add_scalar('Train/smooth', loss_dict['smooth'], global_step)
                    writer.add_scalar('Train/cons', loss_dict['cons'], global_step)
                    writer.add_scalar('Train/diversity', loss_dict.get('diversity', 0.0), global_step)
                    writer.add_scalar('Train/stage1_bgfg_kl', loss_dict.get('stage1_bgfg_kl', 0.0), global_step)
                    writer.add_scalar('Train/stage1_bg_mean', loss_dict.get('stage1_bg_mean', 0.0), global_step)
                    writer.add_scalar('Train/stage1_fg_mean', loss_dict.get('stage1_fg_mean', 0.0), global_step)
                    writer.add_scalar('Train/learning_rate', current_lr, global_step)
    
    # 平均损失
    avg_losses = {k: v / len(train_loader) for k, v in total_losses.items()}
    
    return avg_losses


@torch.no_grad()
def validate(model, val_loader, criterion, epoch, config, writer, rank):
    """验证 - 计算所有损失项"""
    model.eval()

    # 检测是否使用Two-Stage DINOSAUR（用于对比损失）
    target_model = model.module if hasattr(model, 'module') else model
    use_two_stage = hasattr(target_model, 'dinosaur') and target_model.dinosaur.__class__.__name__ == 'TwoStageDINOSAURpp'

    # 验证也使用同一套 warmup 权重（便于对比当前训练目标）
    _apply_loss_warmup(criterion, config, epoch)
    _maybe_force_featrec_only(criterion, config, epoch, rank, writer)
    
    total_losses = {
        'total': 0.0,
        'feat_rec': 0.0,
        'compact': 0.0,
        'entropy': 0.0,
        'min_usage': 0.0,
        'smooth': 0.0,
        'cons': 0.0,
        'diversity': 0.0,
    }
    
    if rank == 0:
        pbar = tqdm(val_loader, desc=f'Validation')
    else:
        pbar = val_loader
    
    with torch.no_grad():
        for batch in pbar:
            xyz = batch['xyz'].cuda()
            rgb = batch['rgb'].cuda()
            
            reconstruction, slots, masks, sp_feats_proj, sampled_coords = model(xyz, rgb)
            
            # 计算所有损失项
            loss, loss_dict = criterion(reconstruction, sp_feats_proj, slots, masks, sampled_coords, use_two_stage=use_two_stage)
            
            # 累积所有损失
            for key in total_losses.keys():
                total_losses[key] += loss_dict.get(key, 0.0)
    
    # 计算平均损失
    avg_losses = {key: val / len(val_loader) for key, val in total_losses.items()}
    
    if rank == 0:
        print(f"\n验证损失:")
        print(f"  总损失: {avg_losses['total']:.6f}")
        print(f"  feat_rec: {avg_losses['feat_rec']:.6f}")
        print(f"  compact: {avg_losses['compact']:.6f}")
        print(f"  entropy: {avg_losses['entropy']:.6f}")
        print(f"  min_usage: {avg_losses['min_usage']:.6f}")
        print(f"  smooth: {avg_losses['smooth']:.6f}")
        print(f"  cons: {avg_losses['cons']:.6f}")
        print(f"  diversity: {avg_losses['diversity']:.6f}")
        
        if writer is not None:
            writer.add_scalar('Val/total_loss', avg_losses['total'], epoch)
            writer.add_scalar('Val/feat_rec', avg_losses['feat_rec'], epoch)
            writer.add_scalar('Val/compact', avg_losses['compact'], epoch)
            writer.add_scalar('Val/entropy', avg_losses['entropy'], epoch)
            writer.add_scalar('Val/min_usage', avg_losses['min_usage'], epoch)
            writer.add_scalar('Val/smooth', avg_losses['smooth'], epoch)
            writer.add_scalar('Val/cons', avg_losses['cons'], epoch)
            writer.add_scalar('Val/diversity', avg_losses['diversity'], epoch)
    
    return avg_losses['total']


@torch.no_grad()
def visualize_samples(model, vis_samples, epoch, config, output_dir):
    """可视化样本 - 适配Mask3D输入"""
    import sys
    import signal
    
    print(f"\n[可视化] 开始生成epoch {epoch}的可视化...")
    sys.stdout.flush()
    
    model.eval()
    
    vis_dir = os.path.join(output_dir, f'epoch_{epoch:03d}')
    os.makedirs(vis_dir, exist_ok=True)

    # 可视化在无图形环境/3D scatter 下可能非常慢，增加超时保护，避免训练“卡死”
    vis_cfg = config.get('visualization', {})
    if not vis_cfg.get('enabled', True):
        print("[可视化] visualization.enabled=False，跳过本轮可视化")
        sys.stdout.flush()
        return
    static_timeout_sec = int(vis_cfg.get('static_timeout_sec', 120))  # 每张静态图最多 2 分钟
    html_timeout_sec = int(vis_cfg.get('html_timeout_sec', 120))      # 每个 HTML 最多 2 分钟

    def _run_with_timeout(fn, seconds: int, desc: str):
        # 仅 Linux/Unix 支持 SIGALRM
        if not hasattr(signal, 'SIGALRM'):
            return fn()

        def _timeout_handler(signum, frame):
            raise TimeoutError(f"{desc} 超时({seconds}s)")

        old = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(seconds)
        try:
            return fn()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    
    for idx, sample in enumerate(vis_samples):
        print(f"[可视化] 处理样本 {idx+1}/{len(vis_samples)}...")
        sys.stdout.flush()
        
        xyz = sample['xyz'].unsqueeze(0).cuda()  # (1, N, 3)
        rgb = sample['rgb'].unsqueeze(0).cuda()  # (1, N, 3)
        
        print(f"[可视化] 模型前向传播...")
        sys.stdout.flush()
        reconstruction, slots, masks, sp_feats_proj, sampled_coords = model(xyz, rgb)
        
        # 提取数据用于可视化
        num_superpoints = masks.shape[-1]
        
        # 采样后的超点坐标
        sp_coords_np = sampled_coords[0].cpu().numpy()  # (num_superpoints, 3)
        sp_labels_np = np.arange(num_superpoints)
        
        # 原始点云
        xyz_full = xyz[0].cpu().numpy()  # (N, 3)
        rgb_full = rgb[0].cpu().numpy()  # (N, 3)
        
        # 转换为numpy
        masks_np = masks[0].cpu().numpy()  # (num_slots, num_superpoints)
        recon_np = reconstruction[0].cpu().numpy()
        sp_feats_np = sp_feats_proj[0].cpu().numpy()
        slot_assignments = np.argmax(masks_np, axis=0)  # (num_superpoints,)
        
        # 同步CUDA，确保数据已传输完成
        torch.cuda.synchronize()
        
        # ==================== Two-stage 可视化：stage1(bg/fg) + stage2(fg slots) ====================
        # 对于 TwoStageDINOSAURpp，我们在 dinosaur 模块里缓存了 _vis_stage1_masks/_vis_stage2_masks/_vis_fg_prior/_vis_bg_prior
        dinosaur = getattr(model, 'dinosaur', None)
        has_two_stage_vis = (
            dinosaur is not None and
            hasattr(dinosaur, '_vis_stage1_masks') and
            hasattr(dinosaur, '_vis_stage2_masks') and
            hasattr(dinosaur, '_vis_fg_prior') and
            hasattr(dinosaur, '_vis_bg_prior')
        )

        if has_two_stage_vis:
            try:
                stage1_masks = dinosaur._vis_stage1_masks[0].detach().cpu().numpy()  # (2, M)
                stage2_masks = dinosaur._vis_stage2_masks[0].detach().cpu().numpy()  # (S-1, M)
                fg_prior = dinosaur._vis_fg_prior[0].detach().cpu().numpy()          # (M,)
                bg_prior = dinosaur._vis_bg_prior[0].detach().cpu().numpy()          # (M,)

                # Stage1: bg/fg 两个 slot
                print(f"[可视化] 生成静态图: stage1_bgfg.png...")
                sys.stdout.flush()
                _run_with_timeout(
                    lambda: visualize_slot_assignment(
                        sp_coords_np, sp_labels_np, stage1_masks,
                        os.path.join(vis_dir, f'sample_{idx}_stage1_bgfg.png'),
                        num_slots=2
                    ),
                    static_timeout_sec,
                    "stage1_bgfg.png"
                )

                # Stage2: (S-1) 个前景 slot + 额外 1 个背景槽（放在最后一个 slot，便于区分）
                # - 前景 token：按 stage2 slot argmax 上色
                # - 背景 token：被分到最后一个 slot（颜色固定且不会混进前景 slot）
                stage2_fg = stage2_masks * fg_prior[None, :]  # (S-1, M)
                stage2_vis = np.concatenate([stage2_fg, bg_prior[None, :]], axis=0)  # (S, M)

                print(f"[可视化] 生成静态图: stage2_slots.png...")
                sys.stdout.flush()
                _run_with_timeout(
                    lambda: visualize_slot_assignment(
                        sp_coords_np, sp_labels_np, stage2_vis,
                        os.path.join(vis_dir, f'sample_{idx}_stage2_slots.png'),
                        num_slots=stage2_vis.shape[0]
                    ),
                    static_timeout_sec,
                    "stage2_slots.png"
                )
            except TimeoutError as e:
                print(f"[可视化 WARNING] {e}，跳过 two-stage slot 可视化")
                sys.stdout.flush()
            except Exception as e:
                print(f"[可视化 WARNING] two-stage slot 可视化失败: {e}")
                sys.stdout.flush()
        else:
            # 单阶段：保持原有 slot_assignment 可视化
            print(f"[可视化] 生成静态图: slot_assignment.png...")
            sys.stdout.flush()
            try:
                _run_with_timeout(
                    lambda: visualize_slot_assignment(
                        sp_coords_np, sp_labels_np, masks_np,
                        os.path.join(vis_dir, f'sample_{idx}_slot_assignment.png'),
                        num_slots=config['model']['num_slots']
                    ),
                    static_timeout_sec,
                    "slot_assignment.png"
                )
            except TimeoutError as e:
                print(f"[可视化 WARNING] {e}，跳过该图")
                sys.stdout.flush()
        
        print(f"[可视化] 生成静态图: recon_error.png...")
        sys.stdout.flush()
        try:
            _run_with_timeout(
                lambda: visualize_reconstruction_error(
                    sp_coords_np, sp_labels_np, recon_np, sp_feats_np,
                    os.path.join(vis_dir, f'sample_{idx}_recon_error.png')
                ),
                static_timeout_sec,
                "recon_error.png"
            )
        except TimeoutError as e:
            print(f"[可视化 WARNING] {e}，跳过该图")
            sys.stdout.flush()
        
        print(f"[可视化] 生成静态图: slot_stats.png...")
        sys.stdout.flush()
        try:
            _run_with_timeout(
                lambda: visualize_slot_statistics(
                    masks_np,
                    os.path.join(vis_dir, f'sample_{idx}_slot_stats.png')
                ),
                static_timeout_sec,
                "slot_stats.png"
            )
        except TimeoutError as e:
            print(f"[可视化 WARNING] {e}，跳过该图")
            sys.stdout.flush()
        
        # PLY 导出（slot 分配结果）
        if vis_cfg.get('save_ply', True):
            print(f"[可视化] 导出 PLY（slot 分配）...")
            sys.stdout.flush()
            ply_path = os.path.join(vis_dir, f'sample_{idx}_slot_assignment.ply')
            try:
                # 通过最近邻把点映射到 superpoint
                from scipy.spatial import cKDTree
                tree = cKDTree(sp_coords_np)
                _, nn_idx = tree.query(xyz_full, k=1)
                point_slot = slot_assignments[nn_idx]  # (N,)
                
                # 生成 slot 颜色
                num_slots = int(config['model']['num_slots'])
                if num_slots <= 20:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    cmap = plt.get_cmap('tab20')
                    palette = np.array([cmap(i)[:3] for i in range(num_slots)], dtype=np.float32)
                else:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    cmap = plt.get_cmap('hsv')
                    palette = np.array([cmap(i / num_slots)[:3] for i in range(num_slots)], dtype=np.float32)
                colors = palette[point_slot % num_slots]
                
                # 写入 PLY
                try:
                    import open3d as o3d
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyz_full.astype(np.float64))
                    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
                    o3d.io.write_point_cloud(ply_path, pcd)
                    print(f"[可视化] PLY 已保存: {ply_path}")
                except ImportError:
                    print("[可视化 WARNING] open3d 未安装，跳过 PLY 导出")
            except Exception as e:
                print(f"[可视化 WARNING] PLY 导出失败: {e}")
            sys.stdout.flush()
        
        print(f"[可视化] 样本 {idx+1} 完成")
        sys.stdout.flush()
    
    print(f"✓ 可视化已保存到: {vis_dir}")


def save_checkpoint(epoch, model, optimizer, scheduler, val_loss, config, output_dir, is_best=False):
    """保存checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': {
            'projector': model.projector.state_dict() if hasattr(model, 'projector') else model.module.projector.state_dict(),
            'dinosaur': model.dinosaur.state_dict() if hasattr(model, 'dinosaur') else model.module.dinosaur.state_dict()
        },
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'config': config
    }
    
    # 常规保存
    if epoch % config['checkpoint']['save_interval'] == 0:
        save_path = os.path.join(output_dir, f'epoch_{epoch:03d}.pth')
        torch.save(checkpoint, save_path)
        print(f"✓ Checkpoint已保存: {save_path}")
    
    # 最佳模型
    if is_best:
        best_path = os.path.join(output_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"✓ 最佳模型已保存: {best_path}")


@torch.no_grad()
def export_slot_assignment_ply(model, sample, config, output_path: str):
    """
    用训练好的 DINOSAUR 权重对单个样本推理，并导出 slot 绑定结果为 PLY（按 slot 上色）。

    - 输入点：sample['xyz'], sample['rgb']  (N,3)
    - 模型输出：sampled_coords (M,3), masks (S,M)
    - 点级 slot 绑定：用最近邻把每个原始点映射到最近的 sampled_coords，再取该 superpoint 的 slot id
    """
    try:
        import open3d as o3d
    except Exception as e:
        raise ImportError(f"导出 PLY 需要 open3d，请先安装。原始错误: {e}")

    xyz = sample['xyz'].unsqueeze(0).cuda()  # (1,N,3)
    rgb = sample['rgb'].unsqueeze(0).cuda()  # (1,N,3) in [0,1]

    model.eval()
    reconstruction, slots, masks, sp_feats_proj, sampled_coords = model(xyz, rgb)

    xyz_np = xyz[0].cpu().numpy()
    rgb_np = rgb[0].cpu().numpy()
    sp_coords_np = sampled_coords[0].cpu().numpy()           # (M,3)
    masks_np = masks[0].cpu().numpy()                        # (S,M)
    sp_slot = np.argmax(masks_np, axis=0).astype(np.int64)   # (M,)

    # 通过最近邻把点映射到 superpoint
    tree = cKDTree(sp_coords_np)
    _, nn_idx = tree.query(xyz_np, k=1)
    point_slot = sp_slot[nn_idx]  # (N,)

    # 生成 slot 颜色（tab20 不够就 hsv）
    num_slots = int(config['model']['num_slots'])
    if num_slots <= 20:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('tab20')
        palette = np.array([cmap(i)[:3] for i in range(num_slots)], dtype=np.float32)
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('hsv')
        palette = np.array([cmap(i / num_slots)[:3] for i in range(num_slots)], dtype=np.float32)
    colors = palette[point_slot % num_slots]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_np.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"[Infer] slot 绑定 PLY 已保存: {output_path}")


def main():
    # 解析参数
    parser = argparse.ArgumentParser()
    # 说明：如果你希望完全不依赖命令行参数，可以直接改这里的默认配置路径
    parser.add_argument('--config', type=str, default='config/config_train_concerto_scannet.yaml')
    parser.add_argument('--test_run', action='store_true', help='测试运行模式（仅2个epoch）')
    parser.add_argument('--resume', type=str, default='', help='checkpoint路径，留空则不恢复')
    args = parser.parse_args()
    
    # 加载配置
    config_path = os.path.join(current_dir, args.config)
    config = load_config(config_path)
    
    # GPU绑定：环境变量优先（支持守护脚本外部指定）
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print(f"[Info] 使用环境变量指定的GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        gpu_ids = config.get('gpu_ids')
        if gpu_ids:
            gpu_ids_str = ','.join(str(g) for g in gpu_ids)
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
            print(f"[Info] 使用配置文件指定的GPU: {gpu_ids_str}")

    # === 自动多卡启动（不依赖 torchrun）===
    # 说明：仅设置 CUDA_VISIBLE_DEVICES 并不会启动多卡；DDP 需要多进程 + RANK/WORLD_SIZE 环境变量。
    if 'RANK' not in os.environ and 'WORLD_SIZE' not in os.environ:
        # 优先按 config.gpu_ids 的数量启动；若为空则按当前可见 GPU 数量
        cfg_gpu_ids = config.get('gpu_ids') or []
        desired_world_size = len(cfg_gpu_ids) if len(cfg_gpu_ids) > 0 else torch.cuda.device_count()
        if desired_world_size > 1:
            print(f"[DDP] 检测到多GPU需求(world_size={desired_world_size})，自动使用 spawn 启动多进程 DDP")
            mp.spawn(
                _ddp_spawn_worker,
                nprocs=desired_world_size,
                args=(desired_world_size, args, config),
                join=True
            )
            return

    # 单进程（或 torchrun/ddp worker）直接进入训练逻辑
    _main_impl(args, config)


if __name__ == '__main__':
    main()

