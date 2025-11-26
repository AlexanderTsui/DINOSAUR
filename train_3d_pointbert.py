"""
DINOSAUR 3D训练脚本 (PointBERT特征输入)

训练流程:
1. S3DIS数据加载 → 重采样到8192点
2. PointBERT提取超点特征 (冻结)
3. 特征投影 384→768维
4. DINOSAUR ISA处理
5. 4项损失计算
6. TensorBoard监控
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
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

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

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '../PointBERT'))

# 导入自定义模块
from data.s3dis_dataset_pointbert import S3DISPointBERTDataset, collate_fn_pointbert
from models.pointbert_wrapper import create_pointbert_dinosaur_model
from models.losses import DINOSAURLoss

# 导入可视化工具
sys.path.insert(0, os.path.join(current_dir, 'utils'))
from visualizer import visualize_slot_assignment, visualize_reconstruction_error, visualize_slot_statistics


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
    if not HAS_PLOTLY:
        print("[可视化] 未检测到Plotly，跳过HTML生成")
        return
    
    if original_xyz.size == 0 or slot_xyz.size == 0:
        print("[可视化] 数据为空，跳过HTML生成")
        return
    
    # 原始点云颜色
    orig_colors = _rgb_to_plotly_colors(original_rgb)
    
    # Slot颜色（重复使用调色板）
    color_palette = getattr(plotly_qualitative, 'Dark24', plotly_qualitative.Plotly)
    repeats = (num_slots + len(color_palette) - 1) // len(color_palette)
    expanded_palette = (color_palette * repeats)[:num_slots]
    slot_assignments = np.asarray(slot_assignments, dtype=np.int64)
    slot_point_colors = [expanded_palette[int(idx)] for idx in slot_assignments]
    
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('Original Point Cloud', 'Slot Assignments')
    )
    
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
    
    fig.write_html(output_path)
    print(f"[可视化] HTML已保存: {output_path}")


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


def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, config, writer, rank):
    """训练一个epoch"""
    model.train()
    
    total_losses = {
        'total': 0.0,
        'reconstruction': 0.0,
        'mask_entropy': 0.0,
        'slot_diversity': 0.0,
        'mask_uniformity': 0.0
    }
    
    if rank == 0:
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    else:
        pbar = train_loader
    
    for iter_idx, batch in enumerate(pbar):
        xyzrgb = batch['xyzrgb'].cuda()
        
        optimizer.zero_grad()
        
        # 混合精度前向传播
        with autocast(enabled=config['train']['use_amp']):
            try:
                reconstruction, slots, masks, sp_feats_proj = model(xyzrgb)
                
                # 检查NaN和Inf（更全面的检查）
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
                    print(f"[警告] Epoch {epoch}, Iter {iter_idx}: 检测到NaN/Inf (NaN={has_nan}, Inf={has_inf})，跳过此batch")
                    optimizer.zero_grad()
                    continue
                
                # 计算损失
                loss, loss_dict = criterion(reconstruction, sp_feats_proj, slots, masks)
                
                # 检查loss是否为NaN或Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[警告] Epoch {epoch}, Iter {iter_idx}: Loss异常 (loss={loss.item()})，跳过此batch")
                    optimizer.zero_grad()
                    continue
                
                # 检查loss是否过大（可能是数值不稳定）
                if loss.item() > 1e6:
                    print(f"[警告] Epoch {epoch}, Iter {iter_idx}: Loss过大 (loss={loss.item():.2e})，跳过此batch")
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
            scaler.update()
            continue
        
        # Unscale梯度用于裁剪
        scaler.unscale_(optimizer)
        
        # 检查梯度中的NaN/Inf（在裁剪之前）
        model_params = model.get_trainable_params() if hasattr(model, 'get_trainable_params') else model.module.get_trainable_params()
        has_grad_nan = any(torch.isnan(p.grad).any() if p.grad is not None else False for p in model_params)
        has_grad_inf = any(torch.isinf(p.grad).any() if p.grad is not None else False for p in model_params)
        
        if has_grad_nan or has_grad_inf:
            print(f"[警告] Epoch {epoch}, Iter {iter_idx}: 梯度包含NaN/Inf (NaN={has_grad_nan}, Inf={has_grad_inf})，跳过更新")
            optimizer.zero_grad()
            scaler.update()
            continue
        
        # 梯度裁剪并记录
        try:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model_params,
                config['train']['grad_clip_norm']
            )
        except RuntimeError as e:
            print(f"[错误] Epoch {epoch}, Iter {iter_idx}: 梯度裁剪失败: {e}")
            optimizer.zero_grad()
            scaler.update()
            continue
        
        # 检查梯度异常
        if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm > 100:
            print(f"[警告] Epoch {epoch}, Iter {iter_idx}: 梯度异常 (norm={grad_norm})，跳过更新")
            # 重置scaler状态
            optimizer.zero_grad()
            scaler.update()  # 更新scaler状态
            continue
        
        # 优化器步进
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # 累积损失
        for key in total_losses.keys():
            total_losses[key] += loss_dict[key]
        
        # 日志记录
        if rank == 0:
            if iter_idx % config['log_interval'] == 0:
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{loss_dict['total']:.4f}",
                    'recon': f"{loss_dict['reconstruction']:.4f}",
                    'lr': f"{current_lr:.6f}"
                })
                
                # TensorBoard
                if writer is not None:
                    global_step = epoch * len(train_loader) + iter_idx
                    writer.add_scalar('Train/total_loss', loss_dict['total'], global_step)
                    writer.add_scalar('Train/reconstruction_loss', loss_dict['reconstruction'], global_step)
                    writer.add_scalar('Train/mask_entropy', loss_dict['mask_entropy'], global_step)
                    writer.add_scalar('Train/slot_diversity', loss_dict['slot_diversity'], global_step)
                    writer.add_scalar('Train/mask_uniformity', loss_dict['mask_uniformity'], global_step)
                    writer.add_scalar('Train/learning_rate', current_lr, global_step)
    
    # 平均损失
    avg_losses = {k: v / len(train_loader) for k, v in total_losses.items()}
    
    return avg_losses


@torch.no_grad()
def validate(model, val_loader, criterion, epoch, config, writer, rank):
    """验证 - 计算所有损失项"""
    model.eval()
    
    total_losses = {
        'total': 0.0,
        'reconstruction': 0.0,
        'mask_entropy': 0.0,
        'slot_diversity': 0.0,
        'mask_uniformity': 0.0
    }
    
    if rank == 0:
        pbar = tqdm(val_loader, desc=f'Validation')
    else:
        pbar = val_loader
    
    with torch.no_grad():
        for batch in pbar:
            xyzrgb = batch['xyzrgb'].cuda()
            
            reconstruction, slots, masks, sp_feats_proj = model(xyzrgb)
            
            # 计算所有损失项
            loss, loss_dict = criterion(reconstruction, sp_feats_proj, slots, masks)
            
            # 累积所有损失
            for key in total_losses.keys():
                if key in loss_dict:
                    total_losses[key] += loss_dict[key]
    
    # 计算平均损失
    avg_losses = {key: val / len(val_loader) for key, val in total_losses.items()}
    
    if rank == 0:
        print(f"\n验证损失:")
        print(f"  总损失: {avg_losses['total']:.6f}")
        print(f"  重建损失: {avg_losses['reconstruction']:.6f}")
        print(f"  Mask熵损失: {avg_losses['mask_entropy']:.6f}")
        print(f"  Slot多样性损失: {avg_losses['slot_diversity']:.6f}")
        print(f"  Mask均匀性损失: {avg_losses['mask_uniformity']:.6f}")
        
        if writer is not None:
            writer.add_scalar('Val/total_loss', avg_losses['total'], epoch)
            writer.add_scalar('Val/reconstruction_loss', avg_losses['reconstruction'], epoch)
            writer.add_scalar('Val/mask_entropy_loss', avg_losses['mask_entropy'], epoch)
            writer.add_scalar('Val/slot_diversity_loss', avg_losses['slot_diversity'], epoch)
            writer.add_scalar('Val/mask_uniformity_loss', avg_losses['mask_uniformity'], epoch)
    
    return avg_losses['total']  # 返回总损失用于最佳模型判断


@torch.no_grad()
def visualize_samples(model, vis_samples, epoch, config, output_dir):
    """可视化样本 - 适配PointBERT输入"""
    model.eval()
    
    vis_dir = os.path.join(output_dir, f'epoch_{epoch:03d}')
    os.makedirs(vis_dir, exist_ok=True)
    
    for idx, sample in enumerate(vis_samples):
        xyzrgb = sample['xyzrgb'].unsqueeze(0).cuda()  # (1, 8192, 6)
        
        reconstruction, slots, masks, sp_feats_proj = model(xyzrgb)
        
        # 提取xyz用于可视化（PointBERT输出的超点对应前N个坐标）
        num_superpoints = masks.shape[-1]
        xyz_sample = xyzrgb[0, :num_superpoints, :3].cpu().numpy()
        sp_labels_np = torch.arange(num_superpoints).cpu().numpy()
        
        xyz_full = xyzrgb[0, :, :3].cpu().numpy()
        rgb_full = xyzrgb[0, :, 3:6].cpu().numpy()
        
        # 转换为numpy
        masks_np = masks[0].cpu().numpy()
        recon_np = reconstruction[0].cpu().numpy()
        sp_feats_np = sp_feats_proj[0].cpu().numpy()
        slot_assignments = np.argmax(masks_np, axis=0)
        
        # 可视化
        visualize_slot_assignment(
            xyz_sample, sp_labels_np, masks_np,
            os.path.join(vis_dir, f'sample_{idx}_slot_assignment.png'),
            num_slots=config['model']['num_slots']
        )
        
        visualize_reconstruction_error(
            xyz_sample, sp_labels_np, recon_np, sp_feats_np,
            os.path.join(vis_dir, f'sample_{idx}_recon_error.png')
        )
        
        visualize_slot_statistics(
            masks_np,
            os.path.join(vis_dir, f'sample_{idx}_slot_stats.png')
        )
        
        vis_cfg = config.get('visualization', {})
        if vis_cfg.get('save_html', False):
            html_path = os.path.join(vis_dir, f'sample_{idx}_slot_assignment.html')
            save_slot_assignment_html(
                xyz_full,
                rgb_full,
                xyz_sample,
                slot_assignments,
                config['model']['num_slots'],
                html_path,
                point_size=vis_cfg.get('html_point_size', 2.0),
                slot_size=vis_cfg.get('html_slot_point_size', 4.0)
            )
    
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


def main():
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config_train_pointbert.yaml')
    parser.add_argument('--test_run', action='store_true', help='测试运行模式（仅2个epoch）')
    args = parser.parse_args()
    
    # 加载配置
    config_path = os.path.join(current_dir, args.config)
    config = load_config(config_path)
    
    # GPU绑定（使用配置文件指定的卡）
    gpu_ids = config.get('gpu_ids')
    if gpu_ids:
        gpu_ids_str = ','.join(str(g) for g in gpu_ids)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
        print(f"[Info] 使用GPU: {gpu_ids_str}")
    
    if args.test_run:
        print("\n⚠️  测试运行模式：仅训练2个epoch")
        config['train']['epochs'] = 2
        config['validation']['val_interval'] = 1
        config['checkpoint']['save_interval'] = 1
    
    # 设置随机种子
    torch.manual_seed(config['seed'])
    
    # 分布式设置
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"DINOSAUR 3D训练 (PointBERT特征)")
        print(f"{'='*60}")
        print(f"配置: {args.config}")
        print(f"设备数: {world_size}")
        print(f"Slot数量: {config['model']['num_slots']}")
        print(f"超点数量: {config['model']['pointbert_num_groups']}")
    
    # 创建输出目录
    output_dir = config['checkpoint']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # TensorBoard
    writer = None
    if rank == 0 and HAS_TENSORBOARD:
        log_dir = os.path.join(output_dir, 'logs')
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard日志: {log_dir}")
    
    # 创建数据集
    train_dataset = S3DISPointBERTDataset(
        root_dir=config['data']['s3dis_root'],
        areas=config['data']['train_areas'],
        target_points=config['data']['target_points'],
        augment=True,
        aug_config=config['augmentation']
    )
    
    val_dataset = S3DISPointBERTDataset(
        root_dir=config['data']['s3dis_root'],
        areas=config['data']['val_areas'],
        target_points=config['data']['target_points'],
        augment=False
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size_per_gpu'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        pin_memory=config['train']['pin_memory'],
        collate_fn=collate_fn_pointbert
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn_pointbert
    )
    
    # 准备可视化样本
    vis_samples = [val_dataset[i] for i in range(min(3, len(val_dataset)))]
    
    if rank == 0:
        print(f"\n数据集:")
        print(f"  训练: {len(train_dataset)} 个房间")
        print(f"  验证: {len(val_dataset)} 个房间")
    
    # 创建模型
    model = create_pointbert_dinosaur_model(config).cuda()
    
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=config['distributed']['find_unused_parameters']
        )
    
    # 损失函数
    criterion = DINOSAURLoss(config['loss']['weights'])
    
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
        if current_iter < warmup_iters:
            return current_iter / warmup_iters
        else:
            progress = (current_iter - warmup_iters) / (total_iters - warmup_iters)
            return (1 - progress) ** config['train']['scheduler']['power']
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 混合精度
    scaler = GradScaler(enabled=config['train']['use_amp'])
    
    if rank == 0:
        print(f"\n模型创建完成")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"可训练参数: {trainable_params:,}")
    
    # 训练循环
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(config['train']['epochs']):
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
            
            # 保存最佳模型
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            if rank == 0:
                save_checkpoint(
                    epoch, model.module if world_size > 1 else model,
                    optimizer, scheduler, val_loss, config, output_dir, is_best
                )
        
        # 可视化
        if rank == 0 and (epoch + 1) % config['checkpoint']['save_interval'] == 0:
            visualize_samples(
                model.module if world_size > 1 else model,
                vis_samples, epoch, config,
                os.path.join(output_dir, 'visualizations')
            )
    
    # 训练完成
    total_time = time.time() - start_time
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"训练完成!")
        print(f"总时间: {total_time/3600:.2f} 小时")
        print(f"最佳验证损失: {best_val_loss:.6f}")
        print(f"{'='*60}\n")
        if writer is not None:
            writer.close()
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    # 单机多卡训练：如果未通过torchrun启动，自动重启
    if 'RANK' not in os.environ:
        import shutil
        script_path = os.path.abspath(__file__)
        
        # 从配置文件读取GPU数量
        try:
            config_path = 'config/config_train_pointbert.yaml'
            for i, arg in enumerate(sys.argv[1:]):
                if arg == '--config' and i + 1 < len(sys.argv):
                    config_path = sys.argv[i + 2]
                    break
            config = load_config(os.path.join(current_dir, config_path))
            nproc = len(config.get('gpu_ids', [])) or 'auto'
        except:
            nproc = 'auto'
        
        # 优先使用conda环境中的torchrun（确保使用正确的Python环境）
        conda_env_path = '/home/pbw/data1/.conda/envs/CloudPoints'
        torchrun_path = os.path.join(conda_env_path, 'bin/torchrun')
        
        if os.path.exists(torchrun_path):
            cmd = [torchrun_path, '--standalone', f'--nproc-per-node={nproc}', script_path] + sys.argv[1:]
            os.execv(torchrun_path, cmd)
        else:
            # fallback: 使用conda run（确保使用正确的环境）
            cmd = ['conda', 'run', '-n', 'CloudPoints', 'torchrun', 
                   '--standalone', f'--nproc-per-node={nproc}', script_path] + sys.argv[1:]
            os.execvp('conda', cmd)
    else:
        main()

